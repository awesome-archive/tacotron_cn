from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
import os

import numpy as np
import tensorflow as tf
import scipy.io as sio

from wavenet import WaveNetModel, mu_law_decode, mu_law_encode, skeleton_reader

SAMPLES = 16000
LOGDIR = './logdir'
WINDOW = 25 #1 second of past samples to take into account
WAVENET_PARAMS = './wavenet_params.json'
SAVE_EVERY = None


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    def _ensure_positive_float(f):
        """Ensure argument is a positive float."""
        if float(f) < 0:
            raise argparse.ArgumentTypeError(
                    'Argument must be greater than zero')
        return float(f)

    parser = argparse.ArgumentParser(description='WaveNet motion generation script')
    parser.add_argument(
        'checkpoint', type=str, help='Which model checkpoint to generate from')
    parser.add_argument(
        '--samples',
        type=int,
        default=SAMPLES,
        help='How many frames of motion samples to generate')
    parser.add_argument(
        '--logdir',
        type=str,
        default=LOGDIR,
        help='Directory in which to store the logging '
        'information for TensorBoard.')
    parser.add_argument(
        '--window',
        type=int,
        default=WINDOW,
        help='The number of past samples to take into '
        'account at each step')
    parser.add_argument(
        '--wavenet_params',
        type=str,
        default=WAVENET_PARAMS,
        help='JSON file with the network parameters')
    parser.add_argument(
        '--skeleton_out_path',
        type=str,
        default=None,
        help='Path to output skeleton file')
    parser.add_argument(
        '--fast_generation',
        type=_str_to_bool,
        default=True,
        help='Use fast generation')
    parser.add_argument(
        '--motion_seed',
        type=str,
        default=None,
        help='The skeleton file to start generation from')
    parser.add_argument(
        '--gc_channels',
        type=int,
        default=None,
        help='Number of global condition embedding channels. Omit if no '
             'global conditioning.')
    parser.add_argument(
        '--gc_cardinality',
        type=int,
        default=None,
        help='Number of categories upon which we globally condition.')
    arguments = parser.parse_args()

    return arguments


def create_seed(filename,
                window_size=WINDOW):
    skeleton = np.loadtxt(filename, delimiter=',')
    cut_index = min(skeleton.shape[0], window_size)
    return skeleton, cut_index

def main():
    args = get_arguments()
    started_datestring = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    #logdir is where logging file is saved. different from where generated mat is saved.
    logdir = os.path.join(args.logdir, 'generate', started_datestring)
    with open(args.wavenet_params, 'r') as config_file:
        wavenet_params = json.load(config_file)

    sess = tf.Session()

    input_channels = wavenet_params['input_channels']
    output_channels = wavenet_params['output_channels']
    gt, cut_index = create_seed(args.motion_seed, args.window)
    if np.isnan(np.sum(gt)):
        print('nan detected')
        raise ValueError('NAN detected in seed file')
    seed = tf.constant(gt)

    net = WaveNetModel(
        batch_size=1,
        dilations=wavenet_params['dilations'],
        filter_width=wavenet_params['filter_width'],
        residual_channels=wavenet_params['residual_channels'],
        dilation_channels=wavenet_params['dilation_channels'],
        skip_channels=wavenet_params['skip_channels'],
        use_biases=wavenet_params['use_biases'],
        input_channels=input_channels,
        output_channels=output_channels,
        global_condition_channels=args.gc_channels)

    samples = tf.placeholder(dtype=tf.float32)

    next_sample = net.predict_proba_incremental(samples)
    sess.run(tf.initialize_all_variables())
    sess.run(net.init_ops)
    #TODO: run init_ops only once

    variables_to_restore = {
        var.name[:-2]: var for var in tf.all_variables()
        if not ('state_buffer' in var.name or 'pointer' in var.name)}
    saver = tf.train.Saver(variables_to_restore)

    print('Restoring model from {}'.format(args.checkpoint))
    saver.restore(sess, args.checkpoint)
    if args.motion_seed:
        pass
    else:
        raise ValueError('motion seed not specified!')


    # seed: T x 42 tensor
    # tolist() converts a tf tensor to a list
    gt_list = sess.run(seed).tolist()
    motion = gt_list[:cut_index]
    #motion: list of generated data (along with seed)
    # When using the incremental generation, we need to
    # feed in all priming samples one by one before starting the
    # actual generation.
    # TODO This could be done much more efficiently by passing the waveform
    # to the incremental generator as an optional argument, which would be
    # used to fill the queues initially.
    outputs = [next_sample]
    outputs.extend(net.push_ops)
    #TODO: question: everytime runs next_sample <- predict_proba_incremental(samples), will the q be reinitialized? or just use the queue with elements inserted before?
    print('Priming generation...')
    #for i, x in enumerate(motion[-net.receptive_field: -1]):
    for i, x in enumerate(motion[-net.receptive_field: -2]):
        if i % 10 == 0:
            print('Priming sample {}'.format(i))
        sess.run(outputs, feed_dict={samples: np.reshape(x, (1, input_channels))})
    print('Done.')
    #TODO: check how next_sample <- net.predict_proba_incremental(samples) works. sample is of size 1 x input_channels.
    #TODO: then check if motion[-1] is fed into network twice.
    last_sample_timestamp = datetime.now()
    for step in range(args.samples):
        outputs = [next_sample]
        outputs.extend(net.push_ops)
        window = motion[-1]

        #TODO: why motion[-1] fed into network twice?
        # Run the WaveNet to predict the next sample.
        prediction = sess.run(outputs, feed_dict={samples: np.reshape(window, (1, input_channels))})[0]
        #TODO: next_input = np.concatenate((prediction, gt(4:9)), axis=1). motion.append(next_input)
        motion.append(np.concatenate((np.reshape(prediction, (1, output_channels)), np.reshape(gt_list[cut_index + step][output_channels:], (1, input_channels - output_channels))), axis=1))
        # Show progress only once per second.
        current_sample_timestamp = datetime.now()
        time_since_print = current_sample_timestamp - last_sample_timestamp
        if time_since_print.total_seconds() > 1.:
            print('Sample {:3<d}/{:3<d}'.format(step + 1, args.samples),
                  end='\r')
            last_sample_timestamp = current_sample_timestamp

    print()

    # save result in .mat file
    if args.skeleton_out_path:
        #TODO: save according to Hanbyul rules
        # outdir = os.path.join('logdir','skeleton_generate', os.path.basename(os.path.dirname(args.checkpoint)) + os.path.basename(args.checkpoint)+'window'+str(args.window)+'sample'+str(args.samples))
        outdir_base = os.path.join(args.skeleton_out_path, os.path.basename(os.path.dirname(args.checkpoint)))
        if not os.path.exists(outdir_base):
            os.makedirs(outdir_base)
        scene_name = os.path.basename(os.path.dirname(args.motion_seed))
        scene_dir = os.path.join(outdir_base, scene_name)
        if not os.path.exists(scene_dir):
            os.makedirs(scene_dir)
        filedir = os.path.join(scene_dir, os.path.basename(args.motion_seed))

        motion_array = np.array(motion)
        np.savetxt(filedir, motion_array[:, :output_channels], delimiter=',')
        #sio.savemat(filedir, {'sequence_gt': gt, 'sequence_predict': motion[:, :output_channels], 'global_T': global_T, 'global_Theta': global_Theta,
        #                      'startFrames': startFrames, 'datatype': foldername, 'testdataPath: '})
        print(len(motion))
        print('generated filedir:{0}'.format(filedir))
    print('Finished generating. The result can be viewed in Matlab.')






if __name__ == '__main__':
    main()
