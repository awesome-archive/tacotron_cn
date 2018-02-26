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
    #parser.add_argument(
    #    '--gc_id',
    #    type=float,
    #    default=None,
    #    help='ID of category to generate, if globally conditioned.')
    arguments = parser.parse_args()
    #if arguments.gc_channels is not None:
        #if arguments.gc_cardinality is None:
        #    raise ValueError("Globally conditioning but gc_cardinality not "
        #                     "specified. Use --gc_cardinality=377 for full "
        #                     "VCTK corpus.")

     #   if arguments.gc_id is None:
     #       raise ValueError("Globally conditioning, but global condition was "
     #                         "not specified. Use --gc_id to specify global "
     #                         "condition.")

    return arguments


def create_seed(filename,
                window_size=WINDOW):
    skeleton = np.loadtxt(filename, delimiter=',')
    #gc_id = skeleton[0,-1]
    #skeleton = skeleton[:, :-1]
    cut_index = min(skeleton.shape[0], window_size)
    return skeleton, cut_index#, float(gc_id)

def main():
    args = get_arguments()
    started_datestring = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    #logdir is where logging file is saved. different from where generated mat is saved.
    logdir = os.path.join(args.logdir, 'generate', started_datestring)
    with open(args.wavenet_params, 'r') as config_file:
        wavenet_params = json.load(config_file)

    sess = tf.Session()

    #skeleton_channels = wavenet_params['skeleton_channels']
    input_channels = wavenet_params['input_channels']
    output_channels = wavenet_params['output_channels']
    #gt, cut_index, gc_id = create_seed(os.path.join(args.motion_seed, os.path.basename(args.motion_seed)), args.window)
    gt, cut_index = create_seed(os.path.join(args.motion_seed, os.path.basename(args.motion_seed)), args.window)
    if np.isnan(np.sum(gt)):
        print('nan detected')
        raise ValueError('NAN detected in seed file')
    #if skeleton_channels == 45 or skeleton_channels == 42:
    #    seed = tf.constant(gt[:cut_index, 45 - skeleton_channels:])
    #else:
    #    seed = tf.constant(gt[:cut_index, :])
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
    #todo: Q: how does samples represent T x 42 data? does predict_proba_incremental memorize? A: samples can store multiple frames. T x 42 dim

    if args.fast_generation:
        #next_sample = net.predict_proba_incremental(samples, args.gc_id)
        #next_sample = net.predict_proba_incremental(samples, gc_id)
        next_sample = net.predict_proba_incremental(samples)
    else:
        #next_sample = net.predict_proba(samples, args.gc_id)
        #next_sample = net.predict_proba(samples, gc_id)
        next_sample = net.predict_proba_incremental(samples)

    if args.fast_generation:
        sess.run(tf.initialize_all_variables())
        sess.run(net.init_ops)

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
    #motion[i]: ith frame, list of 42 features
    if args.fast_generation and args.motion_seed:
        # When using the incremental generation, we need to
        # feed in all priming samples one by one before starting the
        # actual generation.
        # TODO This could be done much more efficiently by passing the waveform
        # to the incremental generator as an optional argument, which would be
        # used to fill the queues initially.
        outputs = [next_sample]
        outputs.extend(net.push_ops)

        print('Priming generation...')
        for i, x in enumerate(motion[-net.receptive_field: -1]):
            if i % 10 == 0:
                print('Priming sample {}'.format(i))
            sess.run(outputs, feed_dict={samples: np.reshape(x, (1, input_channels))})
        print('Done.')

    last_sample_timestamp = datetime.now()
    for step in range(args.samples):
        if args.fast_generation:
            outputs = [next_sample]
            outputs.extend(net.push_ops)
            window = motion[-1]
        else:
            if len(motion) > net.receptive_field:
                window = motion[-net.receptive_field:]
            else:
                window = motion
            outputs = [next_sample]

        #TODO: why motion[-1] fed into network twice?
        # Run the WaveNet to predict the next sample.
        prediction = sess.run(outputs, feed_dict={samples: np.reshape(window, (1, input_channels))})[0]
        # prediction = sess.run(outputs, feed_dict={samples: window})[0]
        #TODO: next_input = np.concatenate((prediction, gt(4:9)), axis=1). motion.append(next_input)
        motion.append(np.concatenate((prediction, gt_list[cut_index + step][input_channels:]), axis=1))
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
        outdir = os.path.join(args.skeleton_out_path, os.path.basename(os.path.dirname(args.checkpoint)))
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        filedir = os.path.join(outdir, str(os.path.basename(args.motion_seed)) + '.mat')
        # filedir = os.path.join(outdir, (sub+args.skeleton_out_path))
        sio.savemat(filedir, {'wavenet_predict': motion, 'gt': gt})
        # out = sess.run(decode, feed_dict={samples: motion})
        # todo: write skeleton writer
        # write_skeleton(motion, args.wav_out_path)
        print(len(motion))
        print('generated filedir:{0}'.format(filedir))
    print('Finished generating. The result can be viewed in Matlab.')






if __name__ == '__main__':
    main()
