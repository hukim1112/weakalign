from __future__ import print_function, division
import argparse
import os
from os.path import exists, join, basename
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
#from torch.utils.data import DataLoader
from util.dataloader import DataLoader # modified dataloader
from model.cnn_geometric_model import CNNGeometric, TwoStageCNNGeometric, FeatureCorrelation, featureL2Norm
from model.loss import TransformedGridLoss, WeakInlierCount, TwoStageWeakInlierCount
from data.synth_dataset import SynthDataset
from data.weak_dataset import ImagePairDataset
from data.pf_dataset import PFDataset, PFPascalDataset
from data.download_datasets import download_PF_pascal
from geotnf.transformation import SynthPairTnf,SynthTwoPairTnf,SynthTwoStageTwoPairTnf
from image.normalization import NormalizeImageDict
from util.torch_util import save_checkpoint, str_to_bool
from util.torch_util import BatchTensorToVars
from geotnf.transformation import GeometricTnf
from collections import OrderedDict
import numpy as np
import numpy.random
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
import torch.nn.functional as F
from model.cnn_geometric_model import featureL2Norm
from util.dataloader import default_collate
from util.eval_util import pck_metric, area_metrics, flow_metrics, compute_metric
from options.options import ArgumentParser

print('WeakAlign training script using weak supervision')

# Argument parsing
args,arg_groups = ArgumentParser(mode='train_weak').parse()
print(args)

# Download validation dataset if needed
if args.eval_dataset_path=='' and args.eval_dataset=='pf-pascal':
    args.eval_dataset_path='datasets/proposal-flow-pascal/'
if args.eval_dataset=='pf-pascal' and not exists(args.eval_dataset_path):
    download_PF_pascal(args.eval_dataset_path)

dataset_eval = PFPascalDataset(csv_file=os.path.join(args.eval_dataset_path, 'val_pairs_pf_pascal.csv'),
                      dataset_path=args.eval_dataset_path,
                      transform=NormalizeImageDict(['source_image','target_image']))

dataloader_eval = DataLoader(dataset_eval, batch_size=8,
                        shuffle=False, num_workers=4)
use_cuda = torch.cuda.is_available()
batch_tnf = BatchTensorToVars(use_cuda=use_cuda)

batch = iter(dataloader_eval)

print(batch_tnf(batch))

# for batch_idx, batch in enumerate(dataloader_eval):
# 	tnf_batch = batch_tnf(batch)
# 	print( tnf_batch )

# 	break