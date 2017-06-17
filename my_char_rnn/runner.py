from __future__ import print_function

import json
import sys

import tensorflow as tf

from my_char_rnn.config import _recommended_parameter as param  # just for convenience
from my_char_rnn.data import DataLoader
from my_char_rnn.model import RnnModel

data_loader = DataLoader(param)

print('creating model ...', file=sys.stderr)
model = RnnModel('rnn_model', param, data_loader)
print('model created.', file=sys.stderr)


def train():
    print('start training ...', file=sys.stderr)
    the_learning_rate = param.initial_learning_rate
    saver = tf.train.Saver()

    with tf.Session() as sess:
        try:
            print('loading existing model ...', file=sys.stderr)
            saver.restore(sess, param.saved_session_file_name)
            print('model loaded.', file=sys.stderr)
        except:
            print('no existing model found, initializing variables ...', file=sys.stderr)
            sess.run(tf.global_variables_initializer())
            print('variables initialized.', file=sys.stderr)

        last_state = None
        n_epoch = 50
        save_and_test_every_n_batches = 50
        sched_threshold = 1.0
        decay_after_n_epoch = 10
        for i_epoch in range(n_epoch):
            if i_epoch >= decay_after_n_epoch:
                the_learning_rate *= param.decay_rate

            for i_batch, (input_data, target_data) in enumerate(data_loader.get_training_batches()):
                loss, last_state = model.train(
                    sess,
                    input_data=input_data,
                    target_data=target_data,
                    learning_rate=the_learning_rate,
                    init_state=last_state,
                    sched_threshold=sched_threshold,
                )

                print('epoch {}, batch {}: loss = {:.3f}'.format(i_epoch, i_batch, loss))

                if (i_batch+1)%save_and_test_every_n_batches == 0:
                    saver.save(
                        sess,
                        save_path=param.saved_session_file_name,
                        #global_step=i_epoch*data_loader.n_batches+i_batch,
                    )
                    generate(sess)
                    # plot_weights(last_state)

                    test_input, test_target = data_loader.get_test_data()
                    test_loss = model.test(sess, test_input, test_target)
                    print('Test loss: {}'.format(test_loss))

            sched_threshold *= param.schedule_decay_rate

    print('Training finished.')


def generate(sess=None, file_name=None, save_states=False):
    if sess is None:
        saver = tf.train.Saver()
        with tf.Session() as sess:
            print('Please input the initial text: ', file=sys.stderr)
            initial_text = ''.join(sys.stdin.readlines()).decode('utf8').strip()
            print('Initial text: {}'.format(repr(initial_text)), file=sys.stderr)
            print('generating sample ...', file=sys.stderr)
            saver.restore(sess, param.saved_session_file_name)
            chars, states = do_generate(sess, initial_text)
    else:
        # do_generate(sess, initial_text='what are you doing')
        chars, states = do_generate(sess, initial_text='but')

    print('Generated text:', file=sys.stderr)
    print(''.join(chars))

    if file_name is not None:
        with open(file_name, "w") as f:
            print('sample_chars = ', json.dumps(chars), file=f)
            if save_states:
                # print(states[0])
                json.encoder.FLOAT_REPR = lambda o: '%.3g' % o
                print('sample_states = ', json.dumps([[{'c': x.c[0].tolist(), 'h': x.h[0].tolist()} for x in y] for y in states], indent=None), file=f)


def do_generate(sess, initial_text, temperature = 0.6):
    # return list(zip(*model.sample(sess, initial_text, data_loader, length=1000, temperature=temperature)))
    chars = []
    states = []
    for (char, state) in model.sample(sess, initial_text, data_loader, length=200, temperature=temperature):
        chars.append(char)
        states.append(state)
    return chars, states


# plt.ion()
#
#
# def plot_weights(state):
#     plt.clf()
#
#     plt.subplot(2, 2, 1)
#     plt.title('0c')
#     plt.hist(state[0].c.reshape(-1), bins=10, range=(-100, 100))
#
#     plt.subplot(2, 2, 2)
#     plt.title('0h')
#     plt.hist(state[0].h.reshape(-1), bins=10)
#
#     plt.subplot(2, 2, 3)
#     plt.title('1c')
#     plt.hist(state[1].c.reshape(-1), bins=10, range=(-100, 100))
#
#     plt.subplot(2, 2, 4)
#     plt.title('1h')
#     plt.hist(state[1].h.reshape(-1), bins=10)
#
#     plt.draw()
#     plt.pause(0.001)