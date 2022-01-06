import random
import numpy as np
from collections import OrderedDict

from search.search_spaces import SearchSpace


class OFAMobileNetV3SearchSpace(SearchSpace):

    def __init__(self,
                 image_scale=(192, 256),  # (min img size, max img size)
                 ks_list=(3, 5, 7),  # kernel sizes
                 depth_list=(2, 3, 4),  # depth
                 expand_ratio_list=(3, 4, 6),  # expansion ratio
                 width_mult_list=(1.0, 1.2),  # width multiplier
                 feature_encoding='one-hot', **kwargs):

        super().__init__(**kwargs)

        self.image_scale_list = list(range(image_scale[0], image_scale[1] + 1, 8))
        self.ks_list = list(ks_list)
        self.depth_list = list(depth_list)
        self.expand_ratio_list = list(expand_ratio_list)
        self.width_mult_list = list(width_mult_list)
        self.feature_encoding = feature_encoding

        # upper and lower bound on the decision variables
        self.n_var = 22
        # [image resolution, width mult, layer 1, layer 2, ..., layer 20]
        # image resolution ~ [192, 200, 208, ..., 256]
        # width multiplier ~ [1.0, 1.2]
        # the last two layers in each stage can be skipped
        self.lb = [0] + [0] + [1, 1, 0, 0] + [1, 1, 0, 0] + [1, 1, 0, 0] + [1, 1, 0, 0] + [1, 1, 0, 0]
        self.ub = [len(self.image_scale_list) - 1] + [len(self.width_mult_list) - 1]

        if max(depth_list) == 4:
            self.ub += [int(len(ks_list) * len(expand_ratio_list))] * 20
        elif max(depth_list) == 3:
            self.ub += ([int(len(ks_list) * len(expand_ratio_list))] * 3 + [0]) * 5
        elif max(depth_list) == 2:
            self.ub += ([int(len(ks_list) * len(expand_ratio_list))] * 2 + [0, 0]) * 5
        else:
            ValueError("max depth has to be greater than 2")

        # create the categories for each variable
        self.categories = [list(range(a, b + 1)) for a, b in zip(self.lb, self.ub)]

        # assuming maximum 4 layers per stage for 5 stages
        self.stage_layer_indices = [list(range(2, self.n_var))[i:i + 4]
                                    for i in range(0, 20, 4)]

        # create the mappings between decision variables (genotype) and subnet architectural string (phenotype)
        self.str2var_mapping = OrderedDict()
        self.var2str_mapping = OrderedDict()
        self.str2var_mapping['skip'] = 0
        increment = 1
        for e in self.expand_ratio_list:
            for ks in self.ks_list:
                self.str2var_mapping['ks@{}_e@{}'.format(ks, e)] = increment
                self.var2str_mapping[increment] = (ks, e)
                increment += 1

        print(self.lb)
        print(self.ub)
        print(self.categories)
        print(self.str2var_mapping)
        print(self.var2str_mapping)

    @property
    def name(self):
        return 'ofa_mobilenet_v3'

    def str2var(self, ks, e):
        return self.str2var_mapping['ks@{}_e@{}'.format(ks, e)]

    def var2str(self, v, ub):
        if v > 0:
            return self.var2str_mapping[v]
        else:
            return self.var2str_mapping[random.randint(1, max(ub, 1))]

    def _sample(self,
                distributions: list = None,
                subnet_str=True):

        if distributions:
            # sample from a particular distribution
            # distribution = [(distribution_pointer, shape_params, sse), ...]
            assert len(distributions) == len(self.categories), "one distribution required per variable"

            x = []
            for valid_options, (distribution, params, _) in zip(self.categories, distributions):
                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                try:
                    _x = round(distribution.rvs(
                        *arg, loc=loc, scale=scale) if arg else distribution.rvs(loc=loc, scale=scale))

                    if _x < min(valid_options):
                        _x = min(valid_options)

                    if _x > max(valid_options):
                        _x = max(valid_options)

                except OverflowError:
                    _x = random.choice(valid_options)

                x.append(_x)
        else:
            # uniform random sampling
            x = [random.choice(options) for options in self.categories]

        x = np.array(x).astype(int)

        # repair, in case of skipping the third but not the fourth layer in a stage
        for indices in self.stage_layer_indices:
            if x[indices[-2]] == 0 and x[indices[-1]] > 0:
                x[indices[-2]] = x[indices[-1]]
                x[indices[-1]] = 0

        if subnet_str:
            return self._decode(x)
        else:
            return x

    def _encode(self, subnet_str):
        # a sample subnet string
        # {'r' : 224,
        #  'w' : 1.2,
        #  'ks': [7, 7, 7, 7, 7, 3, 5, 3, 3, 5, 7, 3, 5, 5, 3, 3, 3, 3, 3, 5],
        #  'e' : [4, 6, 4, 6, 6, 6, 6, 6, 3, 4, 4, 4, 6, 4, 4, 3, 3, 6, 3, 4],
        #  'd' : [2, 2, 3, 4, 2]}

        x = [0] * self.n_var
        x[0] = np.where(subnet_str['r'] == np.array(self.image_scale_list))[0][0]
        x[1] = np.where(subnet_str['w'] == np.array(self.width_mult_list))[0][0]

        for indices, d in zip(self.stage_layer_indices, subnet_str['d']):
            for i in range(d):
                idx = indices[i]
                x[idx] = self.str2var(subnet_str['ks'][idx - 2], subnet_str['e'][idx - 2])
        return x

    def _decode(self, x):
        # a sample decision variable vector x corresponding to the above subnet string
        # [(image scale) 4,
        #  (width mult)  0,
        #  (layers)      8, 9, 5, 5, 6, 2, 3, 6, 6, 1, 4, 0, 1, 2, 2, 3, 9, 5, 8, 1]

        ks_list, expand_ratio_list, depth_list = [], [], []
        for indices in self.stage_layer_indices:
            d = len(indices)
            for idx in indices:
                ks, e = self.var2str(x[idx], self.ub[idx])
                ks_list.append(ks)
                expand_ratio_list.append(e)
                if x[idx] < 1:
                    d -= 1
            depth_list.append(d)

        return {
            'r': self.image_scale_list[x[0]],
            'w': self.width_mult_list[x[1]],
            'ks': ks_list, 'e': expand_ratio_list, 'd': depth_list}

    def _features(self, X):
        # X should be a 2D matrix with each row being a decision variable vector
        if self.feat_enc is None:
            # in case the feature encoder is not initialized
            if self.feature_encoding == 'one-hot':
                from sklearn.preprocessing import OneHotEncoder
                self.feat_enc = OneHotEncoder(categories=self.categories).fit(X)
            else:
                raise NotImplementedError

        return self.feat_enc.transform(X).toarray()


if __name__ == '__main__':
    import time
    import json
    import torch
    from supernets.ofa_mbnv3 import GenOFAMobileNetV3

    from search.algorithms.utils import distribution_estimation
    from search.algorithms.evo_nas import EvoNAS

    search_space = OFAMobileNetV3SearchSpace(image_scale=(224, 224), ks_list=(3, ), expand_ratio_list=(3, 4, ), depth_list=(2, 3, ))
    supernet = GenOFAMobileNetV3(
        n_classes=1000, dropout_rate=0, image_scale_list=search_space.image_scale_list,
        width_mult_list=search_space.width_mult_list, ks_list=search_space.ks_list,
        expand_ratio_list=search_space.expand_ratio_list, depth_list=search_space.depth_list)

    # state_dicts = [
    #     torch.load('/home/cseadmin/zhichao/neural-architecture-transfer/'
    #                'pretrained/backbone/ofa_imagenet/ofa_mbv3_d234_e346_k357_w1.0',
    #                map_location='cpu')['state_dict'],
    #     torch.load('/home/cseadmin/zhichao/neural-architecture-transfer/'
    #                'pretrained/backbone/ofa_imagenet/ofa_mbv3_d234_e346_k357_w1.2',
    #                map_location='cpu')['state_dict']]
    #
    # supernet.load_state_dict(state_dicts)

    # model distribution from data

    # archive = json.load(open("/Users/luzhicha/Dropbox/2021/github/neural-architecture-transfer/tmp/"
    #                          "MobileNetV3SearchSpaceNSGANetV2-acc&flops-lgb-n_doe@100-n_iter@8-max_iter@30/"
    #                          "iter_30/archive.json", 'r'))

    # archs = [m['arch'] for m in archive]
    # X = search_space.encode(archs)
    # F = np.array(EvoNAS.get_attributes(archive, _attr='err&flops'))
    #
    # print(X.shape)
    # print(F.shape)
    # from pymoo.factory import get_reference_directions
    # from search.algorithms.utils import reference_direction_survival, rank_n_crowding_survival
    #
    # sur_X = rank_n_crowding_survival(X, F, n_survive=100)
    # print(sur_X)

    # ref_dirs = get_reference_directions("energy", 3, 100, seed=1)
    # sur_X = reference_direction_survival(ref_dirs, X, F, n_survive=100)
    # print(sur_X.shape)

    # start = time.time()
    # distributions = []
    # for j in range(X.shape[1]):
    #     distributions.append(distribution_estimation(X[:, j]))
    # print("time elapsed = {:.2f} mins".format((time.time() - start) / 60))

    # subnet_str = search_space.sample(10, distributions=distributions)
    # print(subnet_str)
    subnet_str = search_space.sample(10)
    print(subnet_str)

    # print(subnet_str)
    # x = search_space.encode(subnet_str)
    # print(x)
    # features = search_space.features(x)
    # print(features)

    for cfg in subnet_str:
        print(cfg)
        supernet.set_active_subnet(w=cfg['w'], ks=cfg['ks'], e=cfg['e'], d=cfg['d'])
        x = search_space.encode([cfg])[0]
        print(x)
        subnet = supernet.get_active_subnet(preserve_weight=True)
        # print(subnet)
