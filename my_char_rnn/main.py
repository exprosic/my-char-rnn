import select
import sys

import tensorflow as tf

from my_char_rnn.config import _recommended_parameter as param  # just for convenience
from my_char_rnn.data import DataLoader
from my_char_rnn.model import RnnModel

if sys.version_info.major != 2:
    raise RuntimeError('Python 2 required')


def main():
    file_name = 'resource/input.txt'
    n_epoch = 50

    data_loader = DataLoader(param, file_name=file_name)
    data_loader.pre_process()

    model = RnnModel(param, data_loader, scope='a_scope_for_weight_reuse')
    the_learning_rate = param.learning_rate
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        the_state = None
        for i_epoch in range(n_epoch):
            for i_batch, (input_data, target_data) in enumerate(data_loader.get_batches()):
                loss, the_state = model.train(sess,
                                              input_data=input_data,
                                              target_data=target_data,
                                              learning_rate=the_learning_rate,
                                              init_state=the_state)
                print('\x1b[33m[Press Enter to sample]\x1b[0m'
                      ' epoch {}, batch {}: loss = {:.3f}'.format(i_epoch, i_batch, loss))

                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    raw_input()
                    print('\x1b[33m------ sample start ------\x1b[0m')
                    print(model.sample(sess, 'The ', data_loader, 200))
                    print('\x1b[33m------- sample end -------\x1b[0m')
                    print('\x1b[33mPress Enter to continue ...\x1b[0m')
                    raw_input()

        the_learning_rate *= param.decay_rate


if __name__ == '__main__':
    main()
