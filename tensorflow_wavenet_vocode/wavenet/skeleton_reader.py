import fnmatch
import os
import threading

import numpy as np
import tensorflow as tf


def find_files(directory, pattern='*'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

#gc_enabled bool variable indicating whether global condition is used
#store at the end of skeleton global condition (distance in our case)
def load_generic_skeleton(directory, gc_enabled):
    '''Generator that yields audio waveforms from the directory.'''
    files = find_files(directory)
    #print('length of file list {0}'.format(len(files)))
    for filename in files:
        skeleton = np.loadtxt(filename, delimiter=',')
        #if not gc_enabled:
        #    category_id = None
        #else:
        #    category_id = skeleton[:, -1]
        #yield skeleton[:,:-1], filename, category_id
        yield skeleton, filename

class SkeletonReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 skeleton_dir,
                 coord,
                 gc_enabled,
                 receptive_field,
                 input_channels,
                 sample_size=None,
                 queue_size=256):
        self.skeleton_dir = skeleton_dir
        self.coord = coord
        self.sample_size = sample_size
        self.receptive_field = receptive_field
        self.gc_enabled = gc_enabled
        self.threads = []
        self.input_channels = input_channels
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        #self.queue = tf.PaddingFIFOQueue(capacity=queue_size,
        #                                 dtypes=['float32'],
        #                                 shapes=[(None, 1)])
        self.queue = tf.RandomShuffleQueue(capacity=queue_size,
                                           min_after_dequeue=128,
                                           dtypes=['float32'],
                                           shapes=[self.receptive_field + self.sample_size, self.input_channels],
                                           seed=1)
        print("randomsufflequeueinput: {0}, {1}, {2}".format(self.receptive_field , self.sample_size, self.input_channels))
        self.enqueue = self.queue.enqueue([self.sample_placeholder])

        if self.gc_enabled:
            self.id_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
            #self.gc_queue = tf.PaddingFIFOQueue(queue_size, ['float32'],
            #                                    shapes=[()])
            self.gc_queue = tf.RandomShuffleQueue(capacity=queue_size,
                                                  min_after_dequeue=128,
                                                  dtypes=['float32'],
                                                  shapes=[1],
                                                  seed=1)
            self.gc_enqueue = self.gc_queue.enqueue([self.id_placeholder])

        # TODO Find a better way to check this.
        # Checking inside the AudioReader's thread makes it hard to terminate
        # the execution of the script, so we do it in the constructor for now.
        if not find_files(skeleton_dir):
            raise ValueError("No skeleton files found in '{}'.".format(skeleton_dir))
        self.gc_category_cardinality = None

    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        return output

    def dequeue_gc(self, num_elements):
        return self.gc_queue.dequeue_many(num_elements)

    def thread_main(self, sess):
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = load_generic_skeleton(self.skeleton_dir, self.gc_enabled)
            #for skeleton, filename, category_id in iterator:
            for skeleton, filename in iterator:
                if self.coord.should_stop():
                    stop = True
                    break
                #why do we need to pad the input with 0s?
                skeleton = np.pad(skeleton, [[self.receptive_field, 0], [0, 0]],
                               'constant')
                #category_id = np.pad(category_id, [self.receptive_field, 0],
                #                     'constant')
                if self.sample_size:
                    # Cut samples into pieces of size receptive_field +
                    # sample_size with receptive_field overlap
                    while len(skeleton) > self.receptive_field + self.sample_size:
                        piece = skeleton[:(self.receptive_field +
                                        self.sample_size), :]
                        #tf.Print(piece, [piece])
                        if not np.isnan(np.sum(piece)):
                            #TODO: enqueue piece shape: (receptive_field + sample_size, skeleton_channels)
                            sess.run(self.enqueue,
                                     feed_dict={self.sample_placeholder: piece})
                            #TODO: enqueue category_id shape: (1)
                            #if self.gc_enabled:
                            #    sess.run(self.gc_enqueue, feed_dict={
                            #        self.id_placeholder: np.reshape(category_id[0],[1])})
                        skeleton = skeleton[self.sample_size:, :]
                        #category_id = category_id[self.sample_size:]
                else:
                    if not np.isnan(np.sum(skeleton)):
                        sess.run(self.enqueue,
                                 feed_dict={self.sample_placeholder: skeleton})
                        #if self.gc_enabled:
                        #    sess.run(self.gc_enqueue,
                        #             feed_dict={self.id_placeholder: category_id[0]})

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
