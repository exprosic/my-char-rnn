import sys
import select
import numpy as np
import tensorflow as tf

from tensorflow.contrib.rnn import BasicLSTMCell, MultiRNNCell

if sys.version_info.major != 2:
    raise RuntimeError('Python 2 required')


class DataSource(object):
    def __init__(self, file_name, batch_size, seq_length, **ignored_args):
        self.file_name = file_name
        self.batch_size = batch_size
        self.seq_length = seq_length

        # pre-process

        self.text = ''.join(open(self.file_name).readlines())  # full text in a str
        self.n_batches = len(self.text) // (self.batch_size * self.seq_length)
        self.vocab = sorted(set(self.text))  # "abcdefg..."
        self.r_vocab = {c: i for i, c in enumerate(self.vocab)}  # {'a': 0, 'b': 1, ...}

    def get_data(self):
        input_text = self.text[:self.n_batches * self.batch_size * self.seq_length]
        output_text = input_text[1:] + input_text[:1]
        return zip(self._text_to_batches(input_text), self._text_to_batches(output_text))

    def _text_to_batches(self, text):
        assert len(text) == self.n_batches * self.batch_size * self.seq_length
        encoded = np.array([self.r_vocab[c] for c in text])  # .dtype=int, .shape=(len(text),)
        encoded_reshaped = encoded.reshape((self.n_batches, self.batch_size, self.seq_length))
        batches = (x.T for x in encoded_reshaped)  # .shape = (n_batches, seq_length, batch_size)
        return batches


class RnnModel(object):
    def __init__(self, batch_size, seq_length, n_layers, rnn_size, vocab_size, scope, **ignored_args):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.rnn_size = rnn_size
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.scope = scope
        self.grad_clip = 5.0

        self.input_data = tf.placeholder(tf.int32, (self.seq_length, self.batch_size))
        self.target_data = tf.placeholder(tf.int32, (self.seq_length, self.batch_size))
        self.embedding = tf.get_variable("embedding", (self.vocab_size, self.rnn_size))
        embedded = tf.nn.embedding_lookup(self.embedding, self.input_data)
        # embedded.shape = (self.seq_length, self.batch_size, self.rnn_size)
        self.softmax_w = tf.get_variable("softmax_w", (self.rnn_size, self.vocab_size))
        self.softmax_b = tf.get_variable("softmax_b", (self.vocab_size,))
        self.learning_rate = tf.placeholder(tf.float32, ())

        cell = MultiRNNCell([BasicLSTMCell(self.rnn_size) for _ in range(self.n_layers)])
        state = self.init_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        logits = []  # .shape = (seq_length, batch_size, vocab_size)

        with tf.variable_scope(self.scope):
            for i in range(self.seq_length):
                output, state = cell(embedded[i], state)  # output.shape = (batch_size, rnn_size)
                logits.append(tf.matmul(output, self.softmax_w) + self.softmax_b)
                tf.get_variable_scope().reuse_variables()

        self.final_state = state
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_data, logits=logits)
        self.cost = tf.reduce_mean(loss)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.grad_clip)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(grads, tvars))

        # sample model

        self.sample_input_char = tf.placeholder(tf.int32)
        embedded = tf.nn.embedding_lookup(self.embedding, tf.reshape(self.sample_input_char, (1,)))
        self.sample_init_state = cell.zero_state(batch_size=1, dtype=tf.float32)
        with tf.variable_scope(self.scope, reuse=True):
            output, self.sample_final_state = cell(embedded, self.sample_init_state)
            logits = tf.matmul(output, self.softmax_w) + self.softmax_b
            self.sample_output_probs = tf.nn.softmax(logits[0])

    @staticmethod
    def _state_feed(placeholder, value):
        return {p: v for placeholder_i, value_i in zip(placeholder, value) for p, v in zip(placeholder_i, value_i)}

    def train(self, sess, input_data, target_data, learning_rate, init_state=None):
        feed = {self.input_data: input_data, self.target_data: target_data, self.learning_rate: learning_rate}
        if init_state:  # continued training, start from last state
            feed.update(RnnModel._state_feed(self.init_state, init_state))
        loss, state, _ = sess.run([self.cost, self.final_state, self.train_op], feed_dict=feed)
        return loss, state

    def sample(self, sess, initial_text, source, length):
        state = probs = None
        result = []
        for i in range(length):
            c = initial_text[i] if i < len(initial_text) else np.random.choice(source.vocab, p=probs)
            result.append(c)
            feed = {self.sample_input_char: source.r_vocab[c]}
            if state:
                feed.update(RnnModel._state_feed(self.sample_init_state, state))
            probs, state = sess.run([self.sample_output_probs, self.sample_final_state], feed)
        return ''.join(result)


def main():
    args = {
        'scope': 'a_scope_for_reusing_cell_weights',
        'file_name': 'resource/input.txt',
        'batch_size': 50,
        'seq_length': 50,
        'n_layers': 2,
        'rnn_size': 128,
        'n_epoch': 50,
        'learning_rate': 0.002,
        'decay_rate': 0.97,
    }

    source = DataSource(**args)
    args['vocab_size'] = len(source.vocab)

    model = RnnModel(**args)
    lr = args['learning_rate']
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        state = None
        for i_epoch in range(args['n_epoch']):
            for i_batch, (input_data, target_data) in enumerate(source.get_data()):
                loss, state = model.train(sess,
                                          input_data=input_data,
                                          target_data=target_data,
                                          learning_rate=lr,
                                          init_state=state)
                print('\x1b[33m[Press Enter to sample]\x1b[0m'
                      ' epoch {}, batch {}: loss = {:.3f}'.format(i_epoch, i_batch, loss))

                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    raw_input()
                    print('\x1b[33m------ sample start ------\x1b[0m')
                    print(model.sample(sess, 'The ', source, 200))
                    print('\x1b[33m------- sample end -------\x1b[0m')
                    print('\x1b[33mPress Enter to continue ...\x1b[0m')
                    raw_input()

        lr *= args['decay_rate']


if __name__ == '__main__':
    main()
