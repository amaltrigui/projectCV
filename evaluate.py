"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import pathlib
from argparse import ArgumentParser

import h5py
import numpy as np
from runstats import Statistics
from skimage.measure import compare_psnr, compare_ssim


#each output or target is an image or an image volume (tensor with shape C x H x W ?)
def mse(target, output):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((target - output) ** 2)


def nmse(target, output):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(target - output) ** 2 / np.linalg.norm(target) ** 2


def psnr(target, output):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return compare_psnr(target, output, data_range=target.max())  
    # data_range in numerator 
    #Changed in version 0.16: This function was renamed from skimage.measure.compare_psnr to skimage.metrics.peak_signal_noise_ratio



def ssim(target, output):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return compare_ssim( target, output, channel_axis=0, data_range=target.max())
    # default values for k1=0.01;k2=0.03,window_size=7 
    #Changed in version 0.16: This function was renamed from skimage.measure.compare_ssim to skimage.metrics.structural_similarity

def l1(target, output):
    """ Compute Mean Absolute Error (L1). """
    return np.mean(np.abs(target-output))





class Metrics:
    """git 
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        ''' create a Statisitc object for each metric function: 
        a dict with keys=keys of the metric_funcs and values=the Statisitc objects
        metric_funcs = dict(
        MSE=mse,
        NMSE=nmse,
        PSNR=psnr,
        SSIM=ssim,
        L1=l1,               )'''
        self.metric_stats= {
            metric: Statistics() for metric in metric_funcs
        }
        self.metric_funcs= metric_funcs 

    def push(self, target, output):
        ''' add the loss value computed with the given target and output
         values to the corresponding metric statistic object  '''
        for metric, func in self.metric_funcs.items():
            self.metric_stats[metric].push(func(target, output))

    def means(self):
        ''' store the mean values of different losses '''
        return {
            metric: stat.mean() for metric, stat in self.metric_stats.items()
        }

    def stddevs(self):
        ''' store the standard deviations of different losses '''
        return {
            metric: stat.stddev() for metric, stat in self.metric_stats.items()
        }



#still have to see if we need this
def evaluate(args):
    metric_funcs = dict(
        MSE=mse,
        NMSE=nmse,
        PSNR=psnr,
        SSIM=ssim,
        L1=l1,               )
    metrics = Metrics(metric_funcs)    

    for tgt_file in args.target_path.iterdir():  #iterate over paths of the content of the directory
        # open the target image and the correspondint reconstructed image 
        with h5py.File(tgt_file) as target, h5py.File(args.predictions_path / tgt_file.name) as output:  
            target = target['reconstruction_esc'].value  
            output = output['reconstruction'].value
            metrics.push(target, output) 
    return metrics


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--target-path', type=pathlib.Path, required=True,
                        help='Path to the ground truth data')
    parser.add_argument('--predictions-path', type=pathlib.Path, required=True,
                        help='Path to reconstructions')
   
    
    args = parser.parse_args()

    metrics = evaluate(args)
    print(metrics)
