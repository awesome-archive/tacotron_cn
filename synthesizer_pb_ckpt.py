import io
import numpy as np
import tensorflow as tf
from hparams import hparams
# from librosa import effects
from models import create_model
from utils.text import text_to_sequence
from utils import audio

# from tensorflow.python.framework import graph_util

from vad_detect import vad_check

_pad = 0


def _pad_input(x, length):
    return np.pad(x, (0, length - len(x)), mode='constant', constant_values=_pad)


class Synthesizer:
    def __init__(self):
        self.model = None
        self.wav_output = None
        self.session = None
        self.model_filename = None

        # noinspection PyUnusedLocal,PyTypeChecker
    def load(self, checkpoint_path: object, model_name: object = 'tacotron') -> object:
        print('Constructing model: %s' % model_name)
        self.model_filename = checkpoint_path

        if not checkpoint_path.endswith('.pb'):
            inputs = tf.placeholder(tf.int32, [None, None], 'inputs')
            input_lengths = tf.placeholder(tf.int32, [None], 'input_lengths')
            with tf.variable_scope('model') as scope:
                self.model = create_model(model_name, hparams)
                self.model.initialize(inputs, input_lengths)
                self.wav_output = audio.inv_spectrogram_tensorflow(self.model.linear_outputs)
                # self.wav_output = self.model.linear_outputs[0]

            print('Loading checkpoint: %s' % checkpoint_path)
            # self.session = tf.Session()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
            self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            self.session.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(self.session, checkpoint_path)
        else:
            model_filename = checkpoint_path
            with open(model_filename, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                graph = tf.get_default_graph()
                tf.import_graph_def(graph_def, name='')

            self.inputs = graph.get_tensor_by_name("inputs:0")  # 在训练的时候其实可以自己设置
            self.input_lengths = graph.get_tensor_by_name("input_lengths:0")
            self.wav_output = graph.get_tensor_by_name("model/griffinlim/Squeeze:0")

            print('Loading pb: %s' % model_filename)
            # self.session = tf.Session()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
            self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            self.session.run(tf.global_variables_initializer())


    def synthesize(self, inputs):
        cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
        seq_input = [text_to_sequence(j, cleaner_names) for j in inputs]
        seq_length = [len(j) for j in seq_input]
        max_len = max(seq_length)
        inputs = [_pad_input(j, max_len) for j in seq_input]
        seq = np.stack((x for x in inputs))

        # seq = text_to_sequence(text, cleaner_names)
        if not self.model_filename.endswith('.pb'):
            feed_dict = {
                self.model.inputs: np.asarray(seq, dtype=np.int32),
                self.model.input_lengths: np.asarray(seq_length, dtype=np.int32)
            }
        else:
            feed_dict = {
                self.inputs: np.asarray(seq, dtype=np.int32),
                self.input_lengths: np.asarray(seq_length, dtype=np.int32)
            }

        wav = self.session.run(self.wav_output, feed_dict=feed_dict)

        output = []
        print('wav.shape:', wav.shape)
        for wav_index in range(wav.shape[0]):
            wav_index_temp = audio.inv_preemphasis(wav[wav_index])

            wav_index_temp = wav_index_temp[:audio.find_endpoint(wav_index_temp)]
            # wav_index_temp = vad_check(wav_index_temp, hparams.sample_rate)

            out = io.BytesIO()
            audio.save_wav(wav_index_temp, out)
            output.append(out)
        return output

