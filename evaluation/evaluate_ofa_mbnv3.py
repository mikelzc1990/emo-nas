#!/usr/bin/python
# -*- encoding: utf-8 -*-
import sys
sys.path.insert(0, './')

import os
import csv
import yaml
import logging
import argparse
import itertools

import torch
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics

from evaluation.evaluate import validate
from supernets.ofa_mbnv3 import GenOFAMobileNetV3
from search.search_spaces.ofa_search_space import OFAMobileNetV3SearchSpace


parser = argparse.ArgumentParser(description='Validate Supernet')
parser.add_argument('--dataset', default='cifar10', type=str, metavar='DATASET',
                    help='Name of dataset to train (default: "cifar10")')
parser.add_argument('--data', default='~/datasets', type=str, metavar='DATA',
                    help='Path to the dataset images')
parser.add_argument('--supernet', default='../pretrained/backbone/ofa_imagenet', type=str, metavar='SUPERNET',
                    help='Path to folder containing the supernet pretrained weights')
parser.add_argument('--n-classes', type=int, default=1000, metavar='NC',
                    help='number of classes that the supernet was pretrained on (default: 1000 for ImageNet)')
parser.add_argument('--save', default='.tmp', type=str, metavar='SAVE',
                    help='path to dir for saving results')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logger = logging.getLogger()
logger.addHandler(fh)


def main():

    # set up the ofa_mbnv3 search space hyperparameters (modify if needed)
    args.image_sizes = (192, 224, 256)  # image scale ~[192, ..., 224, ..., 256]
    args.ks_list = (3, 5, 7)  # depth-wise conv kernel size ~[3x3, 5x5, 7x7]
    args.depth_list = (2, 3, 4)  # max # of layers for each stage ~[2, 3, 4]
    args.expand_ratio_list = (3, 4, 6)  # expansion ratio ~[3x, 4x, 6x]
    args.width_mult_list = (1.0,)  # width multiplier ~[1.0x, 1.2x]

    # construct the data provider
    args.workers = 4
    args.valid_size = None
    if args.dataset == 'imagenet':
        from data.data_providers.imagenet import ImagenetDataProvider as DataProvider
        args.train_batch_size = 128
        args.test_batch_size = 100
        args.sub_train_size = 2000
        args.sub_train_batch_size = 200
        args.resize_scale = 0.08

    elif args.dataset == 'cifar10':
        from data.data_providers.cifar import CIFAR10DataProvider as DataProvider
        args.train_batch_size = 96
        args.test_batch_size = 100
        args.sub_train_size = 960
        args.sub_train_batch_size = 96
        args.resize_scale = 1.0

    elif args.dataset == 'cifar100':
        from data.data_providers.cifar import CIFAR100DataProvider as DataProvider
        args.train_batch_size = 96
        args.test_batch_size = 100
        args.sub_train_size = 960
        args.sub_train_batch_size = 96
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
        args.resize_scale = 1.0

    else:
        raise NotImplementedError

    # Cache the args as a text string to save them in the output dir
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    data_provider = DataProvider(
        args.data, args.train_batch_size, args.test_batch_size, args.valid_size,
        n_worker=args.workers, resize_scale=args.resize_scale, distort_color=None, image_size=list(args.image_sizes),
        num_replicas=None, rank=None)

    # construct the search space instance
    search_space = OFAMobileNetV3SearchSpace(
        image_scale=args.image_sizes, ks_list=args.ks_list, expand_ratio_list=args.expand_ratio_list,
        depth_list=args.depth_list, width_mult_list=args.width_mult_list)

    # construct the supernet instance
    supernet = GenOFAMobileNetV3(
        n_classes=args.n_classes, dropout_rate=0, image_scale_list=search_space.image_scale_list,
        width_mult_list=search_space.width_mult_list, ks_list=search_space.ks_list,
        expand_ratio_list=search_space.expand_ratio_list, depth_list=search_space.depth_list)

    # define the path to the pretrained supernet weights (in correspondence with width multiplier)
    state_dicts = [torch.load(os.path.join(
        args.supernet, 'ofa_mbv3_d234_e346_k357_w{}'.format(wid_mult)), map_location='cpu')['state_dict']
                   for wid_mult in args.width_mult_list]

    supernet.load_state_dict(state_dicts)

    # define the loss
    criterion = torch.nn.CrossEntropyLoss()

    # create the combinations to validate performance
    wid_mult_options = [min(supernet.width_mult_list), max(supernet.width_mult_list)]
    ks_options = [min(supernet.ks_list), max(supernet.ks_list)]
    expand_ratio_options = [min(supernet.expand_ratio_list), max(supernet.expand_ratio_list)]
    depth_options = [min(supernet.depth_list), max(supernet.depth_list)]

    val_settings = list(itertools.product(
        wid_mult_options, depth_options, expand_ratio_options, ks_options, args.image_sizes))

    # prepare a csv file to store data
    csv_header = ['wid_mult', 'depth', 'expand_ratio', 'kernel_size', 'image_size', 'loss', 'top1', 'top5']

    losses, top1s, top5s, csv_data = [], [], [], []
    for w, d, e, ks, r in val_settings:
        logger.info('w={:.1f}, d={:d}, e={:d}, ks={:d}, r={:d},'.format(r, w, ks, e, d))

        # set image size
        data_provider.assign_active_img_size(r)
        dl = data_provider.test
        sdl = data_provider.build_sub_train_loader(args.sub_train_size, args.sub_train_batch_size)

        # set subnet settings
        supernet.set_active_subnet(w=w, ks=ks, e=e, d=d)
        subnet = supernet.get_active_subnet(preserve_weight=True)

        # reset BN running statistics
        subnet.train()
        set_running_statistics(subnet, sdl)

        # measure acc
        loss, (top1, top5) = validate(subnet, dl, criterion, epoch=0)
        losses.append(loss)
        top1s.append(top1)
        top5s.append(top5)

        # append data to csv
        csv_data.append([w, d, e, ks, r, loss, top1, top5])

    # dump data to csv
    with open(os.path.join(args.save, 'summary.csv'), 'w', encoding='UTF8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(csv_header)
        csv_writer.writerows(csv_data)


if __name__ == "__main__":
    main()