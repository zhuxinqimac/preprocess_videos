import os
import sys
import glob
import argparse
from pipes import quote
from multiprocessing import Pool, current_process

import random
import time

def reverse_list(ls_origin):
    ls_reversed = ls_origin[::-1]
    return ls_reversed

def  reverse_func(vid_item):
    vid_path = vid_item[0]
    vid_id = vid_item[1]
    if dataset == 'UCF101':
        vid_class_name = vid_path.split('/')[-2]
        vid_name = vid_path.split('/')[-1]
        out_full_path = os.path.join(out_path, vid_class_name, vid_name)
    elif dataset == 'Something':
        vid_name = vid_path.split('/')[-1]
        out_full_path = os.path.join(out_path, vid_name)
    try:
        print("Making dir:", out_full_path)
        os.makedirs(out_full_path)
    except OSError:
        pass
    frame_names = sorted(glob.glob(vid_path+'/*'))
    data = [None]*len(frame_names)
    for i, fname in enumerate(frame_names):
        with open(fname, 'rb') as f:
            data[i] = f.read()
    shuffled_names = reverse_list(frame_names)
    for i, fname in enumerate(shuffled_names):
        outname = os.path.join(out_full_path, fname.split('/')[-1])
        with open(outname, 'wb') as f:
            f.write(data[i])
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="shuffle frames")
    parser.add_argument("--origin_path", type=str, 
                        help="origin frame path root")
    parser.add_argument("--out_path", type=str, 
                        help="output path root")
    parser.add_argument("--dataset", type=str, 
                        default="UCF101", 
                        choices=["UCF101", "Something"], 
                        help="dataset type")
    parser.add_argument("--num_worker", type=int, 
                        default=4, 
                        help="number of workers")
    
    args = parser.parse_args()
    origin_path = args.origin_path
    out_path = args.out_path
    dataset = args.dataset
    num_worker = args.num_worker

    if (dataset == 'UCF101'):
        vid_list = glob.glob(origin_path+'/*/*')
    elif (dataset == 'Something'):
        vid_list = glob.glob(origin_path+'/*')
    else:
        print("Dataset should be chosen from [UCF101, Something]")
        raise

    if not os.path.isdir(out_path):
        print("creating folder: "+out_path)
        os.makedirs(out_path)

    print("No. videos:", len(vid_list))
    pool = Pool(num_worker)
    pool.map(reverse_func, zip(vid_list, range(len(vid_list))))
