""" Run NSGANetV2 on OnceForAll MobileNetV3 search space """
import os
import yaml
import pathlib
import argparse

import torch

from search.algorithms.nsganetv2 import MSuNAS
from supernets.ofa_mbnv3 import GenOFAMobileNetV3
from search.evaluators.ofa_evaluator import OFAEvaluator
from search.search_spaces.ofa_search_space import OFAMobileNetV3SearchSpace


parser = argparse.ArgumentParser(description='Warm-up Supernet Training')
parser.add_argument('--dataset', default='cifar10', type=str, metavar='DATASET',
                    help='Name of dataset to train (default: "cifar10")')
parser.add_argument('--data', default='~/datasets', type=str, metavar='DATA',
                    help='Path to the dataset images')
parser.add_argument('--objs', default='acc&flops', type=str, metavar='OBJ',
                    help='which objectives to optimize, separated by "&"')
parser.add_argument('--n-classes', type=int, default=1000, metavar='NC',
                    help='number of classes that the supernet was pretrained on (default: 1000 for ImageNet)')
parser.add_argument('--n-epochs', type=int, default=0, metavar='NE',
                    help='number of epochs to fine-tune subnets before assessing performance')
parser.add_argument('--save', default='.tmp', type=str, metavar='SAVE',
                    help='path to dir for saving results')
parser.add_argument('--resume', default=None, type=str, metavar='RESUME',
                    help='path to dir for resume of search')
args = parser.parse_args()

# search related settings
args.surrogate = 'lgb'  # which surrogate model to fit accuracy predictor
args.n_doe = 100  # design of experiment points, i.e., number of initial (usually randomly sampled) points
args.n_gen = 8  # number of high-fidelity evaluations per generation/iteration
args.max_gens = 30  # maximum number of generations/iterations to search
args.num_subnets = 4  # number of subnets spanning the Pareto front that you would like find


def main():

    # set up the search space hyperparameters
    args.image_sizes = (192, 256)  # image scale ~[192, ..., 224, ..., 256]
    args.ks_list = (3, 5, 7)  # depth-wise conv kernel size ~[3x3, 5x5, 7x7]
    args.depth_list = (2, 3, 4)  # max # of layers for each stage ~[2, 3, 4]
    args.expand_ratio_list = (3, 4, 6)  # expansion ratio ~[3x, 4x, 6x]
    args.width_mult_list = (1.0,)  # width multiplier ~[1.0x, 1.2x]

    # define the path to the pretrained supernet weights (in correspondence with width multiplier)
    state_dicts = [torch.load(os.path.join(
        pathlib.Path(__file__).parent.resolve(), 'pretrained', 'backbone', 'ofa_imagenet',
        'ofa_mbv3_d234_e346_k357_w1.0'), map_location='cpu')['state_dict']]

    # construct the search space instance
    search_space = OFAMobileNetV3SearchSpace(
        image_scale=args.image_sizes, ks_list=args.ks_list, expand_ratio_list=args.expand_ratio_list,
        depth_list=args.depth_list, width_mult_list=args.width_mult_list)

    # construct the supernet instance
    supernet = GenOFAMobileNetV3(
        n_classes=args.n_classes, dropout_rate=0, image_scale_list=search_space.image_scale_list,
        width_mult_list=search_space.width_mult_list, ks_list=search_space.ks_list,
        expand_ratio_list=search_space.expand_ratio_list, depth_list=search_space.depth_list)

    supernet.load_state_dict(state_dicts)

    # construct the data provider
    args.workers = 4
    if args.dataset == 'imagenet':
        from data.data_providers.imagenet import ImagenetDataProvider as DataProvider
        args.train_batch_size = 128
        args.test_batch_size = 100
        args.sub_train_size = 2000
        args.sub_train_batch_size = 200
        args.valid_size = 10000
        args.resize_scale = 0.08

    elif args.dataset == 'cifar10':
        from data.data_providers.cifar import CIFAR10DataProvider as DataProvider
        args.train_batch_size = 96
        args.test_batch_size = 100
        args.sub_train_size = 960
        args.sub_train_batch_size = 96
        args.valid_size = None  # use new CIFAR-10 test set for searching
        args.resize_scale = 1.0

    elif args.dataset == 'cifar100':
        from data.data_providers.cifar import CIFAR100DataProvider as DataProvider
        args.train_batch_size = 96
        args.test_batch_size = 100
        args.sub_train_size = 960
        args.sub_train_batch_size = 96
        args.valid_size = 5000
        args.resize_scale = 1.0

    elif args.dataset == 'food':
        from data.data_providers.food101 import Food101DataProvider as DataProvider
        args.train_batch_size = 128
        args.test_batch_size = 100
        args.sub_train_size = 1280
        args.sub_train_batch_size = 128
        args.valid_size = 5000
        args.resize_scale = 1.0

    elif args.dataset == 'flowers':
        from data.data_providers.flowers102 import Flowers102DataProvider as DataProvider
        args.train_batch_size = 32
        args.test_batch_size = 100
        args.sub_train_size = 320
        args.sub_train_batch_size = 32
        args.valid_size = 2000  # a subset from testing set
        args.resize_scale = 1.0

    else:
        raise NotImplementedError

    # Cache the args as a text string to save them in the output dir
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    data_provider = DataProvider(
        args.data, args.train_batch_size, args.test_batch_size, args.valid_size,
        n_worker=args.workers, resize_scale=args.resize_scale, distort_color=None, image_size=list(args.image_sizes),
        num_replicas=None, rank=None)

    if supernet.n_classes != data_provider.n_classes:
        supernet.reset_classifier(data_provider.n_classes)  # change the task-specific layer accordingly

    # define the evaluator
    evaluator = OFAEvaluator(
        supernet, data_provider, sub_train_size=args.sub_train_size,
        sub_train_batch_size=args.sub_train_batch_size, n_epochs=args.n_epochs)

    # construct MSuNAS search engine
    nas_method = MSuNAS(
        search_space, evaluator, objs=args.objs, surrogate=args.surrogate, n_doe=args.n_doe, n_gen=args.n_gen,
        max_gens=args.max_gens, save_path=args.save, resume=args.resume)

    # kick-off the search
    nas_method.search()

    with open(os.path.join(nas_method.save_path, 'args.yaml'), 'w') as f:
        f.write(args_text)


if __name__ == '__main__':
    main()
