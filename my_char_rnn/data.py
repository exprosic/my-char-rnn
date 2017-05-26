from __future__ import print_function

import os
import sys
import urllib2

import numpy as np

from my_char_rnn.config import ModelParameter


def cache_data(param):
    assert isinstance(param, ModelParameter)

    file_path = os.path.join(param.resource_path, param.data_name)
    try:
        return open(file_path, 'r')
    except IOError:
        pass

    try:
        url = param.input_file_urls[param.data_name]
    except Exception as e:
        raise ValueError('No url for {}'.format(param.data_name))

    print('Downloading {} ...'.format(param.data_name))
    tmp_file_path = file_path+'__'
    with open(tmp_file_path, 'wb') as f:
        f.write(urllib2.urlopen(url).read())
    print('{} downloaded.'.format(param.data_name))

    os.rename(tmp_file_path, file_path)
    return open(file_path, 'r')


class DataLoader(object):
    def __init__(self, param):
        assert isinstance(param, ModelParameter)
        self.param = param

        # Load the full text, calculate the vocabulary and the parameters.

        print('loading data ...', file=sys.stderr)
        self._full_text = ''.join(cache_data(param).readlines()).decode('utf8')  # full text in a str
        print('data loaded.', file=sys.stderr)

        print('preprocessing data ...', file=sys.stderr)
        test_text_length = min(1000, len(self._full_text) // 4)
        self._training_text = self._full_text[:-test_text_length]
        self._test_text = self._full_text[-test_text_length:]

        self.vocab = sorted(set(self._full_text))  # "abcdefg..."
        self.r_vocab = {c: i for i, c in enumerate(self.vocab)}  # {'a': 0, 'b': 1, ...}
        self.n_batches = len(self._training_text) // (self.param.batch_size * self.param.seq_length)
        self.vocab_size = len(self.vocab)
        print('data preprocessed.', file=sys.stderr)

    def get_training_batches(self):
        """
        Batch generator.

        :return: One batch of (input, target) at each iteration.
        """
        full_input_text = self._training_text[:self.n_batches * self.param.batch_size * self.param.seq_length]
        full_output_text = full_input_text[1:] + full_input_text[:1]
        return zip(self._text_to_batches(full_input_text), self._text_to_batches(full_output_text))

    def get_test_data(self):
        input_text = self._test_text
        output_text = self._test_text[1:] + self._test_text[:1]
        return self._encode(input_text), self._encode(output_text)  # .dtype=int, .shape=(len(test_text))

    def _encode(self, text):
        return np.array([self.r_vocab[c] for c in text])

    def _text_to_batches(self, text):
        """
        Translate each character into one-hot encoding vector and reshape.

        :param text: str
        :return: ndarray with shape (n_batches, seq_length, batch_size)
        """
        assert len(text) == self.n_batches * self.param.batch_size * self.param.seq_length
        encoded = self._encode(text)  # .dtype=int, .shape=(len(text),)
        # encoded_reshaped = encoded.reshape((self.n_batches, self.param.batch_size, self.param.seq_length))
        # batches = (x.T for x in encoded_reshaped)  # .shape = (n_batches, seq_length, batch_size)

        encoded_reshaped = encoded.reshape((self.param.batch_size, self.n_batches, self.param.seq_length))
        batches = (encoded_reshaped[:,i,:].T for i in range(self.n_batches))

        # encoded_reshaped = encoded.reshape((self.n_batches * self.param.batch_size, self.param.seq_length))
        # batches = (encoded_reshaped[i:i+self.param.batch_size,:].T for i in range((self.n_batches-1)*self.param.batch_size + 1))
        return batches
