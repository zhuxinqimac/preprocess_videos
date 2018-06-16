
from __future__ import print_function

import os
import sys
import glob
import argparse
from pipes import quote
from multiprocessing import Pool, current_process


def run_optical_flow(vid_item):
    vid_path = vid_item[0]
    vid_id = vid_item[1]
    vid_class_name = vid_path.split('/')[-2]
    vid_name = vid_path.split('/')[-1]
    out_full_path = os.path.join(flow_path, vid_class_name, vid_name)
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass

    current = current_process()
    dev_id = (int(current._identity[0]) - 1) % NUM_GPU
    # flow_x_path = '{}/flow_x'.format(out_full_path)
    # flow_y_path = '{}/flow_y'.format(out_full_path)

    # cmd = os.path.join(df_path + 'build/extract_gpu')+' -f {} -x {} -y {} -i {} -b 20 -t 1 -d {} -s 1 -o {} -w {} -h {}'.format(
        # quote(vid_path), quote(flow_x_path), quote(flow_y_path), quote(image_path), dev_id, out_format, new_size[0], new_size[1])
    cmd = os.path.join(df_path, 'flow_video ')+'-p {} -o {} {}/image_%05d.jpg'.format(
            quote(proc_type), quote(out_full_path), quote(vid_path))

    os.system(cmd)
    print('{} {} done'.format(vid_id, vid_name))
    sys.stdout.flush()
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="extract optical flows")
    parser.add_argument("--frame_path", type=str, default='./UCF101_frames',
                        help='path to the video data')
    parser.add_argument("--flow_path", type=str, default='./UCF101_flow/',
                        help='path to the output flow dir')
    parser.add_argument("--df_path", type=str, default='./flow_toolbox/',
                                    help='path to the flow toolbox')

    parser.add_argument("--proc_type", type=str, default='gpu', choices=['gpu','cpu'],
                        help='choose process type')
    parser.add_argument("--num_worker", type=str, default='8', 
                        help='number of workers')
    NUM_GPU = 1
    args = parser.parse_args()

    flow_path = args.flow_path
    frame_path = args.frame_path
    proc_type = args.proc_type
    num_worker = int(args.num_worker)
    df_path = args.df_path

    if not os.path.isdir(flow_path):
        print("creating folder: "+flow_path)
        os.makedirs(flow_path)

    vid_list = glob.glob(frame_path+'/*/*')
    print('no. videos:', len(vid_list))
    pool = Pool(num_worker)
    pool.map(run_optical_flow, zip(vid_list, range(len(vid_list))))

