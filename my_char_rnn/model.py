import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, MultiRNNCell, DropoutWrapper

from my_char_rnn.config import ModelParameter
from my_char_rnn.data import DataLoader


class RnnModel(object):
    def __init__(self, param, data_loader, scope):
        """

        :param param: model parameters
        :param data_loader: data loader, of which only vocab_size is used here
        :param scope: scope name for weight reusing in the unfolded RNN and between training & testing network
        """
        assert isinstance(param, ModelParameter)
        assert isinstance(data_loader, DataLoader)
        assert isinstance(scope, str)
        self.param = param
        self.scope = scope

        self.input_data = tf.placeholder(tf.int32, (param.seq_length, param.batch_size))
        self.target_data = tf.placeholder(tf.int32, (param.seq_length, param.batch_size))
        self.embedding = tf.get_variable("embedding", (data_loader.vocab_size, param.rnn_size))
        embedded = tf.nn.embedding_lookup(self.embedding, self.input_data)
        # embedded.shape = (self.seq_length, self.batch_size, self.rnn_size)
        self.softmax_w = tf.get_variable("softmax_w", (param.rnn_size, data_loader.vocab_size))
        self.softmax_b = tf.get_variable("softmax_b", (data_loader.vocab_size,))
        self.learning_rate = tf.placeholder(tf.float32, ())

        basic_cells = [BasicLSTMCell(param.rnn_size) for _ in range(param.n_layers)]
        cell = MultiRNNCell([DropoutWrapper(
            x,
            input_keep_prob=param.input_keep_prob,
            output_keep_prob=param.output_keep_prob
        ) for x in basic_cells])

        cell_sample = MultiRNNCell(basic_cells)

        state = self.init_state = cell.zero_state(batch_size=param.batch_size, dtype=tf.float32)
        logits = []  # .shape = (seq_length, batch_size, vocab_size)

        self.sched_threshold = tf.placeholder(tf.float32, ())
        with tf.variable_scope(self.scope) as the_scope:
            probs = None
            for i in range(param.seq_length):
                if i == 0:
                    embedded_input = embedded[i]
                else:
                    random_value = tf.random_uniform(())
                    input_chars = tf.argmax(probs, 1)
                    embedded_input = tf.nn.embedding_lookup(self.embedding, input_chars)
                    embedded_input = tf.cond(
                        random_value>self.sched_threshold,
                        lambda: embedded_input,
                        lambda: embedded[i],
                    )
                output, state = cell(embedded[i], state)  # output.shape = (batch_size, rnn_size)
                logit = tf.matmul(output, self.softmax_w) + self.softmax_b
                logits.append(logit)
                probs = tf.nn.softmax(logit)

                the_scope.reuse_variables()

        self.final_state = state
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_data, logits=logits)
        self.cost = tf.reduce_mean(loss)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), param.grad_clip)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(grads, tvars))

        # sample model

        self.sample_input_char = tf.placeholder(tf.int32)  # [0, vocab_size)
        self.sample_target_char = tf.placeholder(tf.int32)  # [0, vocab_size)
        embedded = tf.nn.embedding_lookup(self.embedding, tf.reshape(self.sample_input_char, (1,)))
        self.sample_init_state = cell_sample.zero_state(batch_size=1, dtype=tf.float32)
        self.temperature = tf.placeholder(tf.float32, ())

        with tf.variable_scope(self.scope, reuse=True):
            output, self.sample_final_state = cell_sample(embedded, self.sample_init_state)
            logits = tf.matmul(output, self.softmax_w) + self.softmax_b
            self.sample_output_probs = tf.nn.softmax(logits[0]/self.temperature)
            self.sample_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(self.sample_target_char, shape=(1,)),
                logits=logits
            )

    @staticmethod
    def _state_feed(placeholder, value):
        return {p: v for placeholder_i, value_i in zip(placeholder, value) for p, v in zip(placeholder_i, value_i)}

    def train(self, sess, input_data, target_data, learning_rate, init_state, sched_threshold):
        feed = {
            self.input_data: input_data,
            self.target_data: target_data,
            self.learning_rate: learning_rate,
            self.sched_threshold: sched_threshold,
        }
        if init_state:  # continued training, start from last state
            feed.update(RnnModel._state_feed(self.init_state, init_state))
        loss, state, _ = sess.run([self.cost, self.final_state, self.train_op], feed_dict=feed)
        return loss, state

    def test(self, sess, input_data, target_data):
        last_state = None
        loss = 0.0
        for input_data_i, target_data_i in zip(input_data, target_data):
            feed = {self.sample_input_char: input_data_i, self.sample_target_char: target_data_i}
            loss += sess.run(self.sample_loss, feed)
        loss /= len(input_data)
        return loss

    def sample(self, sess, initial_text, data_loader, length, temperature):
        last_state = probs = None
        # result = []
        for i in range(length):
            c = initial_text[i] if i < len(initial_text) else np.random.choice(data_loader.vocab, p=probs)
            # result.append(c)
            feed = {self.sample_input_char: data_loader.r_vocab[c], self.temperature: temperature}
            if last_state:
                feed.update(RnnModel._state_feed(self.sample_init_state, last_state))
            probs, last_state = sess.run([self.sample_output_probs, self.sample_final_state], feed)
            yield (c, last_state)
