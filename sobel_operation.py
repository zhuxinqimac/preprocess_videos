import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import numpy as np
import tensorflow as tf
import skimage.io

import argparse
import glob
from multiprocessing import Pool, current_process

def proc_frames(frames_item):
    (in_path, proc_id) = frames_item
    class_name = in_path.split('/')[-2]
    vid_name = in_path.split('/')[-1]
    out_dir = os.path.join(OUT_PATH, class_name, vid_name)
    try:
        os.makedirs(out_dir)
    except:
        pass

    # Get frame data
    print(in_path)
    frame_names = glob.glob(in_path+'/*')
    data = [None]*len(frame_names)
    for i, name in enumerate(frame_names):
       data[i] = skimage.io.imread(name)
    data = np.array(data)
    assert len(data.shape) == 4

    # Get sobeled data
    print('getting sobel')
    # with tf.Session() as sess:
    feed_dict = {input_plh:data}
    im_data = np.array(sess.run(out_data, feed_dict=feed_dict))
    im_data = np.squeeze(im_data, axis=-1)
    for i, f_data in enumerate(im_data):
        full_name = os.path.join(out_dir, frame_names[i].split('/')[-1])
        skimage.io.imsave(full_name, f_data)
    print('{} {} done'.format(proc_id, vid_name))
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Do Sobel Operation")
    parser.add_argument("--in_path", type=str, default='./UCF101',
                        help='path to the video frame data')
    parser.add_argument("--out_path", type=str, default='./UCF101_sobeled/',
                        help='path to the output sobel dir')
    parser.add_argument("--num_worker", type=int, default=2)
    args = parser.parse_args()

    IN_PATH = args.in_path
    OUT_PATH = args.out_path
    num_worker = args.num_worker

    sess = tf.InteractiveSession()

    input_plh = tf.placeholder(dtype=tf.float32, 
                    shape=(None, None, None, 3))
    # kernel_h = [[1,0,-1], [2,0,-2], [1,0,-1]]
    # kernel_h_tf = tf.constant(kernel_h, shape=[1,3,3,1], dtype=tf.float32)
    # kernel_v = [[1,2,1], [0,0,0], [-1,-2,-1]]
    # kernel_v_tf = tf.constant(kernel_v, shape=[1,3,3,1], dtype=tf.float32)
    kernel_v_tf = tf.tile(tf.constant([[1,2,1],[0,0,0],[-1,-2,-1]],
                    shape=[3,3,1,1],dtype=tf.float32),[1,1,3,1])
    kernel_h_tf = tf.transpose(kernel_v_tf,[1,0,2,3])
    grad_x = tf.nn.conv2d(input_plh, kernel_h_tf, 
                    [1,1,1,1], padding='SAME')
    grad_y = tf.nn.conv2d(input_plh, kernel_v_tf, 
                    [1,1,1,1], padding='SAME')

    grad = tf.sqrt(tf.add(tf.pow(grad_x, 2), tf.pow(grad_y, 2)))
    # out_data = grad
    grad = tf.clip_by_value(grad, 0., tf.reduce_max(grad))
    out_data = tf.truediv(grad, tf.reduce_max(grad))

    vid_list = glob.glob(IN_PATH+'/*/*')
    # pool = Pool(num_worker)
    # pool.map(proc_frames, zip(vid_list, range(len(vid_list))))
    for i in vid_list:
        proc_frames((i, 0))

