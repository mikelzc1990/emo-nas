import sys
sys.path.insert(0, './')

import copy
import time
import warnings
from abc import ABC
from tqdm import tqdm

import torch
import torch.nn as nn

from torchprofile import profile_macs
from timm.scheduler import CosineLRScheduler
from cutmix.utils import CutMixCrossEntropyLoss

from ofa.utils import get_net_device
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics
from ofa.imagenet_classification.data_providers.base_provider import DataProvider

from evaluation.utils import *
from evaluation.evaluate import validate
from supernets.ofa_mbnv3 import GenOFAMobileNetV3


class OFAEvaluator(ABC):
    def __init__(self,
                 supernet: GenOFAMobileNetV3,
                 data_provider: DataProvider,  # data provider class
                 sub_train_size=2000,  # number of images to calibrate BN stats
                 sub_train_batch_size=200,  # batch size for subset train dataloader
                 # training related settings
                 n_epochs=0,  # number of training epochs
                 lr=0.025,  # initial learning rate
                 momentum=0.9,
                 wd=3e-4,  # weight decay
                 grad_clip=5,  # gradient clipping
                 ):
        self.supernet = supernet
        self.data_provider = data_provider
        self.num_classes = data_provider.n_classes
        self.sub_train_size = sub_train_size
        self.sub_train_batch_size = sub_train_batch_size

        self.n_epochs = n_epochs
        if n_epochs:
            # applying few epochs training before evaluating a subnet's performance
            self.lr = lr
            self.momentum = momentum
            self.wd = wd
            self.grad_clip = grad_clip

        self.criterion = torch.nn.CrossEntropyLoss()

    @staticmethod
    def _calc_params(subnet):
        return sum(p.numel() for p in subnet.parameters() if p.requires_grad) / 1e6  # in unit of Million

    @staticmethod
    def _calc_flops(subnet, dummy_data):
        dummy_data = dummy_data.to(get_net_device(subnet))
        return profile_macs(subnet, dummy_data) / 1e6  # in unit of MFLOPs

    @staticmethod
    def measure_latency(subnet, input_size, iterations=None):
        """ Be aware that latency will fluctuate depending on the hardware operating condition,
        e.g., loading, temperature, etc. """

        print("measuring latency....")

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        subnet.eval()
        model = subnet.cuda()
        input = torch.randn(*input_size).cuda()

        with torch.no_grad():
            for _ in range(10):
                model(input)

            if iterations is None:
                elapsed_time = 0
                iterations = 100
                while elapsed_time < 1:
                    torch.cuda.synchronize()
                    torch.cuda.synchronize()
                    t_start = time.time()
                    for _ in range(iterations):
                        model(input)
                    torch.cuda.synchronize()
                    torch.cuda.synchronize()
                    elapsed_time = time.time() - t_start
                    iterations *= 2
                FPS = iterations / elapsed_time
                iterations = int(FPS * 6)

            print('=========Speed Testing=========')
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            t_start = time.time()
            for _ in tqdm(range(iterations)):
                model(input)
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            elapsed_time = time.time() - t_start
            latency = elapsed_time / iterations * 1000
        torch.cuda.empty_cache()
        # FPS = 1000 / latency (in ms)
        return latency

    def _measure_latency(self, subnet, input_size):
        return self.measure_latency(subnet, input_size)

    @staticmethod
    def train_one_epoch(epoch, subnet, dataloader, optimizer, criterion, scheduler,
                        grad_clip=None, run_str='', no_logs=False):
        subnet.train()  # switch to training mode

        n_batch = len(dataloader)
        losses = AverageMeter()

        with tqdm(total=len(dataloader),
                  desc='Training Epoch #{} {}'.format(epoch + 1, run_str), disable=no_logs) as t:

            num_updates = epoch * n_batch
            for step, (x, target) in enumerate(dataloader):
                x, target = x.cuda(), target.cuda(non_blocking=True)
                logits = subnet(x)
                loss = criterion(logits, target)

                losses.update(loss.item(), x.size(0))
                optimizer.zero_grad()
                loss.backward()

                if grad_clip:
                    nn.utils.clip_grad_norm_(subnet.parameters(), grad_clip)

                optimizer.step()

                torch.cuda.synchronize()
                num_updates += 1

                lrl = [param_group['lr'] for param_group in optimizer.param_groups]
                lr = sum(lrl) / len(lrl)

                scheduler.step_update(num_updates=num_updates, metric=losses.avg)

                t.set_postfix({
                    'loss': losses.avg,
                    'lr': lr,
                    'img_size': x.size(2),
                })
                t.update(1)

    def train(self, subnet, image_size, **kwargs):
        # train network model for some epochs before assessing performance

        # create a non-dynamic training data loader
        train_loader = self.data_provider.__class__(
            save_path=self.data_provider.save_path, train_batch_size=self.data_provider.train_batch_size,
            test_batch_size=self.data_provider.test_batch_size, valid_size=self.data_provider.valid_size,
            n_worker=self.data_provider.n_worker, resize_scale=self.data_provider.resize_scale, image_size=image_size
        ).train

        optimizer = torch.optim.SGD(subnet.parameters(), self.lr, momentum=self.momentum, weight_decay=self.wd)
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=self.n_epochs,
            lr_min=0.0,
            warmup_lr_init=0.0001,
            warmup_t=1,
            k_decay=1.0,
            **{'cycle_mul': 1., 'cycle_decay': 0.1, 'cycle_limit': 1},
        )
        criterion = CutMixCrossEntropyLoss()  # assuming using cutmix data augmentation

        for epoch in range(self.n_epochs):
            self.train_one_epoch(epoch, subnet, train_loader, optimizer, criterion,
                                 scheduler, self.grad_clip, **kwargs)

            scheduler.step(epoch + 1)

    def eval_acc(self, subnet, dl, sdl, criterion):

        # reset BN running statistics
        subnet.train()
        set_running_statistics(subnet, sdl)
        # measure acc
        loss, (top1, top5) = validate(subnet, dl, criterion)
        return loss, top1, top5

    def evaluate(self, _subnets, objs='acc&flops&params&latency', print_progress=True):
        """ high-fidelity evaluation by inference on validation data """

        subnets = copy.deepcopy(_subnets)  # make a copy of the archs to be evaluated
        batch_stats = []

        for i, subnet_str in enumerate(subnets):
            if print_progress:
                print("evaluating subnet {}:".format(i))
                print(subnet_str)

            stats = {}
            # set subnet accordingly
            image_scale = subnet_str.pop('r')
            input_size = (1, 3, image_scale, image_scale)

            # create dummy data for measuring flops
            dummy_data = torch.rand(*input_size)

            self.supernet.set_active_subnet(**subnet_str)
            subnet = self.supernet.get_active_subnet(preserve_weight=True)
            subnet.cuda()

            # in case a few epochs of training are required
            if self.n_epochs > 0:
                self.train(subnet, image_size=image_scale, no_logs=not print_progress)

            print_str = ''
            if 'acc' in objs:
                # set the image scale
                self.data_provider.assign_active_img_size(image_scale)
                dl = self.data_provider.valid
                sdl = self.data_provider.build_sub_train_loader(self.sub_train_size, self.sub_train_batch_size)

                # compute top-1 accuracy
                _, top1, _ = self.eval_acc(subnet, dl, sdl, self.criterion)

                # batch_acc.append(top1)
                stats['acc'] = top1
                print_str += 'Top1 = {:.2f}'.format(top1)

            # calculate #params and #flops
            if 'params' in objs:
                params = self._calc_params(subnet)
                # batch_params.append(params)
                stats['params'] = params
                print_str += ', #Params = {:.2f}M'.format(params)

            if 'flops' in objs:
                with warnings.catch_warnings():  # ignore warnings, use w/ caution
                    warnings.simplefilter("ignore")
                    flops = self._calc_flops(subnet, dummy_data)
                # batch_flops.append(flops)
                stats['flops'] = flops
                print_str += ', #FLOPs = {:.2f}M'.format(flops)

            if 'latency' in objs:
                latency = self._measure_latency(subnet, input_size)
                # batch_latency.append(latency)
                stats['latency'] = latency
                print_str += ', FPS = {:d}'.format(int(1000 / latency))

            if print_progress:
                print(print_str)
            batch_stats.append(stats)

        return batch_stats


class ImageNetEvaluator(OFAEvaluator):
    def __init__(self,
                 supernet: GenOFAMobileNetV3,
                 data_root='../data',  # path to the data folder
                 valid_isze=10000,  # this is a random subset from train used to guide search
                 batchsize=200, n_workers=4,
                 # following two are for BN running stats calibration
                 sub_train_size=2000, sub_train_batch_size=200):

        # build ImageNet dataset and dataloader
        from data_providers.imagenet import ImagenetDataProvider

        imagenet_dataprovider = ImagenetDataProvider(
            save_path=data_root, train_batch_size=batchsize, test_batch_size=batchsize,
            valid_size=valid_isze, n_worker=n_workers)

        super().__init__(supernet, imagenet_dataprovider, num_classes=1000,
                         sub_train_size=sub_train_size, sub_train_batch_size=sub_train_batch_size)


if __name__ == '__main__':
    from search.search_spaces.ofa_search_space import OFAMobileNetV3SearchSpace

    # construct the supernet
    search_space = OFAMobileNetV3SearchSpace()

    ofa_network = GenOFAMobileNetV3(
        n_classes=1000, dropout_rate=0, image_scale_list=search_space.image_scale_list,
        width_mult_list=search_space.width_mult_list, ks_list=search_space.ks_list,
        expand_ratio_list=search_space.expand_ratio_list, depth_list=search_space.depth_list)

    # load checkpoints weights
    state_dicts = [
        torch.load('/home/cseadmin/zhichao/neural-architecture-transfer/'
                   'pretrained/backbone/ofa_imagenet/ofa_mbv3_d234_e346_k357_w1.0',
                   map_location='cpu')['state_dict'],
        torch.load('/home/cseadmin/zhichao/neural-architecture-transfer/'
                   'pretrained/backbone/ofa_imagenet/ofa_mbv3_d234_e346_k357_w1.2',
                   map_location='cpu')['state_dict']]
    ofa_network.load_state_dict(state_dicts)

    # define the dataset
    from data_providers.cifar import CIFAR10DataProvider as DataProvider
    data_provider = DataProvider(save_path='/home/cseadmin/datasets/')

    # reset classification layer if needed
    if ofa_network.n_classes != data_provider.n_classes:
        ofa_network.reset_classifier(data_provider.n_classes)

    # define the evaluator
    evaluator = OFAEvaluator(ofa_network, data_provider, sub_train_size=960, sub_train_batch_size=96, n_epochs=5)

    archs = search_space.sample(5)
    batch_stats = evaluator.evaluate(archs, objs='acc&flops&params')
    print(archs)
    print(batch_stats)
