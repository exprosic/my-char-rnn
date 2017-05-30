from collections import namedtuple

import tensorflow as tf

from .config import ModelParameter

_default_initializer = tf.random_uniform_initializer(-0.08, 0.08)


class _LinearParam(object):
    def __init__(self, name, input_size, output_size):
        with tf.variable_scope(name):
            self.w = tf.get_variable(
                "w", shape=(input_size, output_size), dtype=tf.float32, initializer=_default_initializer
            )
            self.b = tf.get_variable(
                "b", shape=(output_size,), dtype=tf.float32, initializer=_default_initializer
            )

    def __call__(self, x):
        return tf.matmul(x, self.w) + self.b


class LSTMCell(object):
    class _State(namedtuple("LSTMStateTuple", ["c", "h"])):
        def __init__(self, c, h):
            """
            Dummy constructor for type hint
            :param c: tensor/ndarray/...
            :param h: tensor/ndarray/...
            """
            super(LSTMCell._State, self).__init__(c, h)

        def feed(self, tensor):
            return {self.c: tensor.c, self.h: tensor.h}

    def __init__(self, name, param):
        self.param = param
        with tf.variable_scope(name):
            self.lin_i = _LinearParam("linear_function_i", 2 * param.rnn_size, param.rnn_size)
            self.lin_f = _LinearParam("linear_function_f", 2 * param.rnn_size, param.rnn_size)
            self.lin_o = _LinearParam("linear_function_o", 2 * param.rnn_size, param.rnn_size)
            self.lin_g = _LinearParam("linear_function_g", 2 * param.rnn_size, param.rnn_size)

    def __call__(self, prev_state, input_state):
        prev_and_input = tf.concat([prev_state.h, input_state], 1)
        i = tf.sigmoid(self.lin_i(prev_and_input))
        f = tf.sigmoid(self.lin_f(prev_and_input))
        o = tf.sigmoid(self.lin_o(prev_and_input))
        g = tf.tanh(self.lin_g(prev_and_input))
        new_c = f * input_state + i * g
        new_h = o * tf.tanh(new_c)
        return LSTMCell._State(c=new_c, h=new_h)

    def zero_state(self, batch_size):
        return LSTMCell._State(
            c=tf.zeros((batch_size, self.param.rnn_size)),
            h=tf.zeros((batch_size, self.param.rnn_size))
        )


class MultiLSTMCell(object):
    class _State(list):
        def feed(self, tensors):
            result = {}
            for (node, tensor) in zip(self, tensors):
                result.update(node.feed(tensor))
            return result

    def __init__(self, name, param):
        assert isinstance(param, ModelParameter)
        with tf.variable_scope(name):
            self.cells = MultiLSTMCell._State(LSTMCell("lstm_{}".format(i), param) for i in range(param.n_layers))

    def __call__(self, prev_states, input_states):
        """

        :param prev_states: [].c.shape = [].h.shape = (batch_size, rnn_size)
        :param input_states: [].shape = (batch_size, rnn_size)
        :return: (output, [State])
        """
        result_states = []
        current_input = input_states
        for (prev,cell) in zip(prev_states, self.cells):
            new_state = cell(prev, current_input)
            result_states.append(new_state)
            current_input = new_state.h
        return (current_input, result_states)

    def zero_state(self, batch_size):
        return MultiLSTMCell._State(cell.zero_state(batch_size) for cell in self.cells)
