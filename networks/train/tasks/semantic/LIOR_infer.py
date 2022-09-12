#!/usr/bin/env python3

import argparse
#from msilib import sequence
import subprocess
import datetime
from traceback import print_tb
import yaml
from shutil import copyfile
import os
import shutil
import __init__ as booger
import glob
import numpy as np
import torch
import torch.nn.functional as F
from common.laserscan import LaserScan
import time

def key_func(x):
    return os.path.split(x)[-1]


if __name__ == '__main__':
    
    splits = ["train", "valid", "test"]
    parser = argparse.ArgumentParser("./DROR_infer.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Dataset to test with. No Default',
    )
    
    parser.add_argument(
        '--data_cfg', '-dc',
        type=str,
        required=True,
        default=None,
        help='Directory to get the data config.'
    )

    parser.add_argument(
        '--split', '-s',
        type=str,
        required=False,
        default=None,
        help='Split to evaluate on. One of ' +
             str(splits) + '. Defaults to %(default)s',
    )

    parser.add_argument(
        '--predictions', '-p',
        type=str,
        required=True,
        help='Directory of the predictions.',
    )

    FLAGS, unparsed = parser.parse_known_args()

    print("----------")
    print("INTERFACE:")
    print("dataset", FLAGS.dataset)
    print("data_cfg", FLAGS.data_cfg)
    print("infering", FLAGS.split)
    print("predictions", FLAGS.predictions)
    print("----------")

    # open arch config file
    try:
        print("Opening data config file from %s" % FLAGS.data_cfg)
        CONFIG = yaml.safe_load(open(FLAGS.data_cfg, 'r'))
    except Exception as e:
        print(e)
        print("Error opening arch yaml file.")
        quit()
    
    sequences = CONFIG["split"][FLAGS.split]
    
    alpha = 0.16
    beta = 3.0
    k_min = 3
    sr_min = 0.04
    i_constant = 126455
    i2_constant = 0.0469

    search = 5
    search_dim = search ** 2
    pad = int((search - 1) / 2)

    Scan = LaserScan(project=False, flip_sign=False, H=64, W=2048, fov_up=3.0, fov_down=-25.0)

    for seq in sequences:
        sequence_in = str(seq).zfill(2)
        scans = sorted(glob.glob(FLAGS.dataset + 'sequences/{0}/snow_velodyne/*.bin'.format(sequence_in)), key=key_func)
        for scan in range(len(scans)):
            Scan.open_scan(scans[scan])
            points = Scan.points
            intensities = Scan.remissions
            points = np.swapaxes(points, 0, 1)
            proj_xyz = torch.from_numpy(points.reshape(3,64,2048))
            intensities = torch.from_numpy(intensities).unsqueeze(dim=0)
            proj_range = torch.linalg.norm(proj_xyz, dim=0, keepdim=True)
            
            proj_range = proj_range[None,:,:,:]
            inputs = proj_xyz[None,:,:,:]
                     
            unfold_inputs = F.unfold(inputs,
                                kernel_size=(search, search),
                                padding=(pad, pad))

            x_points = unfold_inputs[:, 0*search_dim:1*search_dim, :]
            y_points = unfold_inputs[:, 1*search_dim:2*search_dim, :]
            z_points = unfold_inputs[:, 2*search_dim:3*search_dim, :]
            xyz_points = torch.cat((x_points, y_points, z_points), dim=0)
            
            proj_xyz = torch.swapaxes(proj_xyz[None,:,:,:].flatten(start_dim=2), 0, 1)
            differences = xyz_points - proj_xyz
            differences = torch.linalg.norm(differences, dim=0)
            differences = differences[None,:,:]

            proj_range = proj_range.flatten(start_dim=2)
            sr_p_map = torch.zeros((proj_range.shape))
            sr_p_map[proj_range < sr_min] = sr_min
            sr_p_map[proj_range >= sr_min] = beta * (proj_range[proj_range >= sr_min] * alpha)

            radius_inliers = torch.count_nonzero(differences < sr_p_map, dim=1)
            predictions = torch.zeros((radius_inliers.shape))
            predictions[radius_inliers < k_min] = 1
            intensity_threshold = i2_constant*(i_constant/proj_range.squeeze(dim=0)**2)
            predictions[intensities > intensity_threshold] = 0
            
            print("sequence: ", seq, "index: ", scan, "/", len(scans), "snow count: ", np.count_nonzero(predictions))
            
            pred_np = predictions.detach().numpy()
            lower_half = pred_np.astype(np.uint16)
            upper_half = np.zeros((pred_np.shape[0])).astype(np.uint16)
            label = (upper_half << 16) + lower_half
            label = np.asarray(label)
            label = label.astype(np.uint32)

            path = os.path.join(FLAGS.predictions, "sequences",
                                sequence_in, "predictions", str(scan).zfill(6) + ".label")
            label.tofile(path)