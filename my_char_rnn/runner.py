import matplotlib.pyplot as plt
import tensorflow as tf

from my_char_rnn.config import _recommended_parameter as param  # just for convenience
from my_char_rnn.data import DataLoader
from my_char_rnn.model import RnnModel

data_loader = DataLoader(param)
model = RnnModel(param, data_loader, scope='a_scope_for_weight_reuse')


def train():
    the_learning_rate = param.initial_learning_rate
    saver = tf.train.Saver()

    with tf.Session() as sess:
        try:
            saver.restore(sess, param.saved_session_file_name)
        except:
            sess.run(tf.global_variables_initializer())

        last_state = None
        n_epoch = 50
        save_and_test_every_n_batches = 50
        sched_threshold = 1.0
        for i_epoch in range(n_epoch):
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
                    plot_weights(last_state)

                    # test_input, test_target = data_loader.get_test_data()
                    # test_loss = model.test(sess, test_input, test_target)
                    # print('Test loss: {}'.format(test_loss))

        the_learning_rate *= param.decay_rate
        sched_threshold *= param.schedule_decay_rate

    print('Training finished.')


def generate(sess):
    if sess is None:
        with tf.Session() as sess:
            print('Please input the initial text: ')
            initial_text = raw_input()
            saver = tf.train.Saver()
            saver.restore(sess, param.saved_session_file_name)
            do_generate(sess, initial_text)
    else:
        do_generate(sess, initial_text='what are you doing')


def do_generate(sess, initial_text):
    generate_text = model.sample(sess, initial_text, data_loader, length=200)
    print('Generated text:')
    print(generate_text)

plt.ion()


def plot_weights(state):
    plt.clf()

    plt.subplot(2, 2, 1)
    plt.title('0c')
    plt.hist(state[0].c.reshape(-1), bins=10, range=(-100, 100))

    plt.subplot(2, 2, 2)
    plt.title('0h')
    plt.hist(state[0].h.reshape(-1), bins=10)

    plt.subplot(2, 2, 3)
    plt.title('1c')
    plt.hist(state[1].c.reshape(-1), bins=10, range=(-100, 100))

    plt.subplot(2, 2, 4)
    plt.title('1h')
    plt.hist(state[1].h.reshape(-1), bins=10)

    plt.draw()
    plt.pause(0.001)