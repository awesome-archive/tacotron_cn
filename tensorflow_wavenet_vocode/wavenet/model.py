import numpy as np
import tensorflow as tf

from .ops import causal_conv, mu_law_encode


def create_variable(name, shape):
    '''Create a convolution filter variable with the specified name and shape,
    and initialize it using Xavier initialition.'''
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable


def create_embedding_table(name, shape):
    if shape[0] == shape[1]:
        # Make a one-hot encoding as the initial value.
        initial_val = np.identity(n=shape[0], dtype=np.float32)
        return tf.Variable(initial_val, name=name)
    else:
        return create_variable(name, shape)


def create_bias_variable(name, shape):
    '''Create a bias variable with the specified name and shape and initialize
    it to zero.'''
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name)


class WaveNetModel(object):
    '''Implements the WaveNet network for generative audio.

    Usage (with the architecture as in the DeepMind paper):
        dilations = [2**i for i in range(N)] * M
        filter_width = 2  # Convolutions just use 2 samples.
        residual_channels = 16  # Not specified in the paper.
        dilation_channels = 32  # Not specified in the paper.
        skip_channels = 16      # Not specified in the paper.
        net = WaveNetModel(batch_size, dilations, filter_width,
                           residual_channels, dilation_channels,
                           skip_channels)
        loss = net.loss(input_batch)
    '''

    def __init__(self,
                 batch_size,
                 dilations,
                 filter_width,
                 residual_channels,
                 dilation_channels,
                 skip_channels,
                 input_channels,
                 output_channels,
                 use_biases=False,
                 histograms=False,
                 global_condition_channels=None):
        '''Initializes the WaveNet model.

        Args:
            batch_size: How many audio files are supplied per batch
                (recommended: 1).
            dilations: A list with the dilation factor for each layer.
            filter_width: The samples that are included in each convolution,
                after dilating.
            residual_channels: How many filters to learn for the residual.
            dilation_channels: How many filters to learn for the dilated
                convolution.
            skip_channels: How many filters to learn that contribute to the
                quantized softmax output.
            quantization_channels: How many amplitude values to use for audio
                quantization and the corresponding one-hot encoding.
                Default: 256 (8-bit quantization).
            use_biases: Whether to add a bias layer to each convolution.
                Default: False.
            histograms: Whether to store histograms in the summary.
                Default: False.
            global_condition_channels: Number of channels in (embedding
                size) of global conditioning vector. None indicates there is
                no global conditioning.
            global_condition_cardinality: Number of mutually exclusive
                categories to be embedded in global condition embedding. If
                not None, then this implies that global_condition tensor
                specifies an integer selecting which of the N global condition
                categories, where N = global_condition_cardinality. If None,
                then the global_condition tensor is regarded as a vector which
                must have dimension global_condition_channels.

        '''
        self.batch_size = batch_size
        self.dilations = dilations
        self.filter_width = filter_width
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.use_biases = use_biases
        self.skip_channels = skip_channels
        self.histograms = histograms
        self.global_condition_channels = global_condition_channels


        self.receptive_field = WaveNetModel.calculate_receptive_field(
            self.filter_width, self.dilations)
        self.variables = self._create_variables()

    @staticmethod
    def calculate_receptive_field(filter_width, dilations):
        receptive_field = (filter_width - 1) * sum(dilations) + 1
        receptive_field += filter_width - 1
        return receptive_field

    def _create_variables(self):
        #TODO: class function start with _.
        #TODO: creates all variables
        '''This function creates all variables used by the network.
        This allows us to share them between multiple calls to the loss
        function and generation function.'''

        var = dict()
        #TODO: var: a dictionary of dictionary. first key: name of block of network. second key: layer name of the block

        with tf.variable_scope('wavenet'):
            with tf.variable_scope('causal_layer'):
                layer = dict()
                initial_channels = self.input_channels
                initial_filter_width = self.filter_width
                #TODO: causal layer filter size: 2 x input_channels x residual_channels
                layer['filter'] = create_variable(
                    'filter',
                    [initial_filter_width,
                     initial_channels,
                     self.residual_channels])
                var['causal_layer'] = layer

            var['dilated_stack'] = list()
            with tf.variable_scope('dilated_stack'):
                for i, dilation in enumerate(self.dilations):
                    #TODO: where does dilation come into play?
                    with tf.variable_scope('layer{}'.format(i)):
                        #TODO: seems like every dilation layer only has one filter of size 2 x residual_channels(size of last layer) x dilation_channels (size of output layer)?
                        current = dict()
                        current['filter'] = create_variable(
                            'filter',
                            [self.filter_width,
                             self.residual_channels,
                             self.dilation_channels])
                        current['gate'] = create_variable(
                            'gate',
                            [self.filter_width,
                             self.residual_channels,
                             self.dilation_channels])
                        #TODO: ['dilated_stack']['dense'] transforms output dimension from dilation_channels to residual channels?
                        current['dense'] = create_variable(
                            'dense',
                            [1,
                             self.dilation_channels,
                             self.residual_channels])
                        #TODO: ['dilated_stack']['skip'] transforms output dimension from dilated_channels to skip_channels?
                        current['skip'] = create_variable(
                            'skip',
                            [1,
                             self.dilation_channels,
                             self.skip_channels])

                        if self.global_condition_channels is not None:
                            current['gc_gateweights'] = create_variable(
                                'gc_gate',
                                [1, self.global_condition_channels,
                                 self.dilation_channels])
                            current['gc_filtweights'] = create_variable(
                                'gc_filter',
                                [1, self.global_condition_channels,
                                 self.dilation_channels])

                        if self.use_biases:
                            current['filter_bias'] = create_bias_variable(
                                'filter_bias',
                                [self.dilation_channels])
                            current['gate_bias'] = create_bias_variable(
                                'gate_bias',
                                [self.dilation_channels])
                            current['dense_bias'] = create_bias_variable(
                                'dense_bias',
                                [self.residual_channels])
                            current['skip_bias'] = create_bias_variable(
                                'slip_bias',
                                [self.skip_channels])

                        var['dilated_stack'].append(current)

            with tf.variable_scope('postprocessing'):
                #TODO: postprocess1: 1x1 conv from skip_channels to skip_channels. (fully connected)
                #TODO: postprocess2: 1x1 conv from skip_channels to skeleton_channels. (fully connected to output size)
                current = dict()
                current['postprocess1'] = create_variable(
                    'postprocess1',
                    [1, self.skip_channels, self.skip_channels])
                current['postprocess2'] = create_variable(
                    'postprocess2',
                    [1, self.skip_channels, self.output_channels])
                if self.use_biases:
                    current['postprocess1_bias'] = create_bias_variable(
                        'postprocess1_bias',
                        [self.skip_channels])
                    current['postprocess2_bias'] = create_bias_variable(
                        'postprocess2_bias',
                        [self.output_channels])
                var['postprocessing'] = current

        return var

    def _create_causal_layer(self, input_batch):
        #TODO: input_batch: N x T x skeleton_channels
        '''Creates a single causal convolution layer.

        The layer can change the number of channels.
        '''
        #TODO: causal layer filter size: 2 x skeleton_channels x residual_channels
        with tf.name_scope('causal_layer'):
            weights_filter = self.variables['causal_layer']['filter']
            return causal_conv(input_batch, weights_filter, 1)
        #TODO: causal_conv : restored = tf.nn.conv1d(value, filter_, stride=1, padding='VALID')
        #TODO: return shape: N x (T-1)? x residual_channels
    def _create_dilation_layer(self, input_batch, layer_index, dilation,
                               global_condition_batch, output_width):
        '''Creates a single causal dilated convolution layer.

        Args:
             input_batch: Input to the dilation layer.
             layer_index: Integer indicating which layer this is.
             dilation: Integer specifying the dilation size.
             global_conditioning_batch: Tensor containing the global data upon
                 which the output is to be conditioned upon. Shape:
                 [batch size, 1, channels]. The 1 is for the axis
                 corresponding to time so that the result is broadcast to
                 all time steps.

        The layer contains a gated filter that connects to dense output
        and to a skip connection:

               |-> [gate]   -|        |-> 1x1 conv -> skip output
               |             |-> (*) -|
        input -|-> [filter] -|        |-> 1x1 conv -|
               |                                    |-> (+) -> dense output
               |------------------------------------|

        Where `[gate]` and `[filter]` are causal convolutions with a
        non-linear activation at the output. Biases and global conditioning
        are omitted due to the limits of ASCII art.

        '''
        variables = self.variables['dilated_stack'][layer_index]
        #TODO: weights_filter.shape = 2 x residual_channels x dilation_channels
        weights_filter = variables['filter']
        weights_gate = variables['gate']

        conv_filter = causal_conv(input_batch, weights_filter, dilation)
        conv_gate = causal_conv(input_batch, weights_gate, dilation)

        if global_condition_batch is not None:
            weights_gc_filter = variables['gc_filtweights']
            conv_filter = conv_filter + tf.nn.conv1d(global_condition_batch,
                                                     weights_gc_filter,
                                                     stride=1,
                                                     padding="SAME",
                                                     name="gc_filter")
            weights_gc_gate = variables['gc_gateweights']
            conv_gate = conv_gate + tf.nn.conv1d(global_condition_batch,
                                                 weights_gc_gate,
                                                 stride=1,
                                                 padding="SAME",
                                                 name="gc_gate")

        if self.use_biases:
            filter_bias = variables['filter_bias']
            gate_bias = variables['gate_bias']
            conv_filter = tf.add(conv_filter, filter_bias)
            conv_gate = tf.add(conv_gate, gate_bias)

        out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)

        # The 1x1 conv to produce the residual output
        weights_dense = variables['dense']
        transformed = tf.nn.conv1d(
            out, weights_dense, stride=1, padding="SAME", name="dense")

        # The 1x1 conv to produce the skip output
        skip_cut = tf.shape(out)[1] - output_width
        out_skip = tf.slice(out, [0, skip_cut, 0], [-1, -1, -1])
        weights_skip = variables['skip']
        skip_contribution = tf.nn.conv1d(
            out_skip, weights_skip, stride=1, padding="SAME", name="skip")

        if self.use_biases:
            dense_bias = variables['dense_bias']
            skip_bias = variables['skip_bias']
            transformed = transformed + dense_bias
            skip_contribution = skip_contribution + skip_bias

        if self.histograms:
            layer = 'layer{}'.format(layer_index)
            tf.histogram_summary(layer + '_filter', weights_filter)
            tf.histogram_summary(layer + '_gate', weights_gate)
            tf.histogram_summary(layer + '_dense', weights_dense)
            tf.histogram_summary(layer + '_skip', weights_skip)
            if self.use_biases:
                tf.histogram_summary(layer + '_biases_filter', filter_bias)
                tf.histogram_summary(layer + '_biases_gate', gate_bias)
                tf.histogram_summary(layer + '_biases_dense', dense_bias)
                tf.histogram_summary(layer + '_biases_skip', skip_bias)

        input_cut = tf.shape(input_batch)[1] - tf.shape(transformed)[1]
        input_batch = tf.slice(input_batch, [0, input_cut, 0], [-1, -1, -1])

        return skip_contribution, input_batch + transformed

    def _generator_conv(self, input_batch, state_batch, weights):
        '''Perform convolution for a single convolutional processing step.'''
        # TODO generalize to filter_width > 2
        # TODO: _generator_conv: convolution over a pair of (input, state)
        past_weights = weights[0, :, :]
        curr_weights = weights[1, :, :]
        output = tf.matmul(state_batch, past_weights) + tf.matmul(
            input_batch, curr_weights)
        return output

    def _generator_causal_layer(self, input_batch, state_batch):
        with tf.name_scope('causal_layer'):
            weights_filter = self.variables['causal_layer']['filter']
            output = self._generator_conv(
                input_batch, state_batch, weights_filter)
        return output

    def _generator_dilation_layer(self, input_batch, state_batch, layer_index,
                                  dilation, global_condition_batch):
        variables = self.variables['dilated_stack'][layer_index]

        weights_filter = variables['filter']
        weights_gate = variables['gate']
        output_filter = self._generator_conv(
            input_batch, state_batch, weights_filter)
        output_gate = self._generator_conv(
            input_batch, state_batch, weights_gate)

        if global_condition_batch is not None:
            global_condition_batch = tf.reshape(global_condition_batch,
                                                shape=(1, -1))
            weights_gc_filter = variables['gc_filtweights']
            weights_gc_filter = weights_gc_filter[0, :, :]
            output_filter += tf.matmul(global_condition_batch,
                                       weights_gc_filter)
            weights_gc_gate = variables['gc_gateweights']
            weights_gc_gate = weights_gc_gate[0, :, :]
            output_gate += tf.matmul(global_condition_batch,
                                     weights_gc_gate)

        if self.use_biases:
            output_filter = output_filter + variables['filter_bias']
            output_gate = output_gate + variables['gate_bias']

        out = tf.tanh(output_filter) * tf.sigmoid(output_gate)

        weights_dense = variables['dense']
        transformed = tf.matmul(out, weights_dense[0, :, :])
        if self.use_biases:
            transformed = transformed + variables['dense_bias']

        weights_skip = variables['skip']
        skip_contribution = tf.matmul(out, weights_skip[0, :, :])
        if self.use_biases:
            skip_contribution = skip_contribution + variables['skip_bias']

        return skip_contribution, input_batch + transformed

    def _create_network(self, input_batch, global_condition_batch):
        '''Construct the WaveNet network.'''
        outputs = []
        current_layer = input_batch


        #TODO: l1: causal conv layer: N x T x residual_channels
        current_layer = self._create_causal_layer(current_layer)
        #TODO: set output width
        output_width = tf.shape(input_batch)[1] - self.receptive_field + 1

        # Add all defined dilation layers.
        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index)):
                    #TODO: l2: dilation conv layers
                    #TODO: current_layer: recursively updated during feedforward: N x T x residual_channels
                    #TODO: output: records skip connection output of each dilated conv layer: N x T x skip_channels
                    output, current_layer = self._create_dilation_layer(
                        current_layer, layer_index, dilation,
                        global_condition_batch, output_width)
                    outputs.append(output)

        with tf.name_scope('postprocessing'):
            # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
            # postprocess the output.
            #TODO: what's the dimension for w1?
            w1 = self.variables['postprocessing']['postprocess1']
            w2 = self.variables['postprocessing']['postprocess2']
            if self.use_biases:
                b1 = self.variables['postprocessing']['postprocess1_bias']
                b2 = self.variables['postprocessing']['postprocess2_bias']

            if self.histograms:
                tf.histogram_summary('postprocess1_weights', w1)
                tf.histogram_summary('postprocess2_weights', w2)
                if self.use_biases:
                    tf.histogram_summary('postprocess1_biases', b1)
                    tf.histogram_summary('postprocess2_biases', b2)

            # We skip connections from the outputs of each layer, adding them
            # all up here.
            #TODO: python built-in sum sums over elements of an iterable.
            #TODO: skip connections sums over outputs of each dilated layer.
            total = sum(outputs)
            #TODO: skip connection go through relu non-linearity
            transformed1 = tf.nn.relu(total)
            #TODO: 1x1 convolution over transformed1.
            conv1 = tf.nn.conv1d(transformed1, w1, stride=1, padding="SAME")
            if self.use_biases:
                conv1 = tf.add(conv1, b1)
            #TODO: relu non-linearity over conv1
            transformed2 = tf.nn.relu(conv1)
            #TODO: 1x1 convolution over transformed2
            conv2 = tf.nn.conv1d(transformed2, w2, stride=1, padding="SAME")
            if self.use_biases:
                conv2 = tf.add(conv2, b2)
            #TODO: two non-linearity + 1x1 conv over sum of skip connections. (equivalent to two fully connected layers)
        return conv2

    def _create_generator(self, input_batch, global_condition_batch):
        '''Construct an efficient incremental generator.'''
        init_ops = []
        push_ops = []
        outputs = []
        current_layer = input_batch

        q = tf.FIFOQueue(
            1,
            dtypes=tf.float32,
            shapes=(self.batch_size, self.input_channels))
        init = q.enqueue_many(
            tf.zeros((1, self.batch_size, self.input_channels)))

        current_state = q.dequeue()
        push = q.enqueue([current_layer])
        #TODO: define two nodes: cuurent_state:dequeue from q. push: enqueue current_layer to queue.
        #TODO: init_ops: list of nodes for initialization.
        init_ops.append(init)
        #TODO: push_ops: list of nodes for pushing results.
        push_ops.append(push)

        #TODO: _generate_causal_layer: one step feed forward with two input frames
        current_layer = self._generator_causal_layer(
                            current_layer, current_state)

        # Add all defined dilation layers.
        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index)):
                    #TODO: declare a new FIFOQueue for each dilation layer
                    q = tf.FIFOQueue(
                        dilation,
                        dtypes=tf.float32,
                        shapes=(self.batch_size, self.residual_channels))
                    init = q.enqueue_many(
                        tf.zeros((dilation, self.batch_size,
                                  self.residual_channels)))

                    current_state = q.dequeue()
                    push = q.enqueue([current_layer])
                    init_ops.append(init)
                    push_ops.append(push)

                    output, current_layer = self._generator_dilation_layer(
                        current_layer, current_state, layer_index, dilation,
                        global_condition_batch)
                    outputs.append(output)
        #TODO: init_ops: initialize all queues with zeros.
        #TODO: push_ops: push all queues with generated results
        self.init_ops = init_ops
        self.push_ops = push_ops

        with tf.name_scope('postprocessing'):
            variables = self.variables['postprocessing']
            # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
            # postprocess the output.
            w1 = variables['postprocess1']
            w2 = variables['postprocess2']
            if self.use_biases:
                b1 = variables['postprocess1_bias']
                b2 = variables['postprocess2_bias']

            # We skip connections from the outputs of each layer, adding them
            # all up here.
            total = sum(outputs)
            transformed1 = tf.nn.relu(total)

            conv1 = tf.matmul(transformed1, w1[0, :, :])
            if self.use_biases:
                conv1 = conv1 + b1
            transformed2 = tf.nn.relu(conv1)
            conv2 = tf.matmul(transformed2, w2[0, :, :])
            if self.use_biases:
                conv2 = conv2 + b2
        #TODO: outputs the final result of one frame of generation
        return conv2


    def predict_proba_incremental(self, input, global_condition=None,
                                  name='wavenet'):
        '''Computes the probability distribution of the next sample
        incrementally, based on a single sample and all previously passed
        samples.'''
        if self.filter_width > 2:
            raise NotImplementedError("Incremental generation does not "
                                      "support filter_width > 2.")
        with tf.name_scope(name):
            raw_output = self._create_generator(input, global_condition)
            #TODO: _create_generator does all generation. input: 1 x input_channels
            out = tf.reshape(raw_output, [-1, self.output_channels])
            last = tf.slice(
                out,
                [tf.shape(out)[0] - 1, 0],
                [1, self.output_channels])
            return tf.reshape(last, [-1])

    def loss(self,
             input_batch,
             global_condition_batch=None,
             l2_regularization_strength=None,
             name='wavenet'):
        '''Creates a WaveNet network and returns the autoencoding loss.

        The variables are all scoped to the given name.
        '''
        #tf.Print(input_batch, [tf.shape(input_batch)])
        with tf.name_scope(name):
            # We mu-law encode and quantize the input audioform.
            #encoded_input = mu_law_encode(input_batch,
            #                              self.skeleton_channels)

            # Cut off the last sample of network input to preserve causality.
            network_input = input_batch
            network_input_width = tf.shape(network_input)[1] - 1
            network_input = tf.slice(network_input, [0, 0, 0],
                                     [-1, network_input_width, -1])

            #TODO: feedforward is calculated by _create_network function. Check if output dimension could be different from input.
            raw_output = self._create_network(network_input, global_condition_batch)

            with tf.name_scope('loss'):
                # Cut off the samples corresponding to the receptive field
                # for the first predicted sample.
                #tf.Print(input_batch, [input_batch])
                '''
                target_output = tf.slice(
                    tf.reshape(
                        input_batch,
                        [self.batch_size, -1, self.output_channels]),
                    [0, self.receptive_field, 0],
                    #[0, 0, 0],
                    [-1, -1, -1])
                '''
                target_output = tf.slice(
                    tf.reshape(
                        input_batch,
                        [self.batch_size, -1, self.input_channels]),
                    [0, self.receptive_field, 0],
                    [-1, -1, self.output_channels])

                target_output = tf.reshape(target_output,
                                           [-1, self.output_channels])
                prediction = tf.reshape(raw_output,
                                        [-1, self.output_channels])
                #loss = tf.nn.softmax_cross_entropy_with_logits(
                #    prediction,
                #    target_output)
                #reduced_loss = tf.reduce_mean(loss)
                loss = tf.nn.l2_loss(prediction - target_output)
                reduced_loss = loss / tf.cast(tf.shape(prediction)[0], tf.float32)
                tf.scalar_summary('loss', reduced_loss)

                if l2_regularization_strength is None:
                    return reduced_loss
                else:
                    # L2 regularization for all trainable parameters
                    l2_loss = tf.add_n([tf.nn.l2_loss(v)
                                        for v in tf.trainable_variables()
                                        if not('bias' in v.name)])

                    # Add the regularization term to the loss
                    total_loss = (reduced_loss +
                                  l2_regularization_strength * l2_loss)

                    tf.scalar_summary('l2_loss', l2_loss)
                    tf.scalar_summary('total_loss', total_loss)

                    return total_loss
