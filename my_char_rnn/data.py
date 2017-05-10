import numpy as np

from my_char_rnn.config import ModelParameter


class DataLoader(object):
    def __init__(self, param):
        assert isinstance(param, ModelParameter)
        self.param = param

        # Load the full text, calculate the vocabulary and the parameters.

        self._full_text = ''.join(open(param.input_file_name).readlines())  # full text in a str

        test_text_length = min(1000, len(self._full_text) // 4)
        self._training_text = self._full_text[:-test_text_length]
        self._test_text = self._full_text[-test_text_length:]

        self.vocab = sorted(set(self._full_text))  # "abcdefg..."
        self.r_vocab = {c: i for i, c in enumerate(self.vocab)}  # {'a': 0, 'b': 1, ...}
        self.n_batches = len(self._training_text) // (self.param.batch_size * self.param.seq_length)
        self.vocab_size = len(self.vocab)

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
        encoded_reshaped = encoded.reshape((self.n_batches, self.param.batch_size, self.param.seq_length))
        batches = (x.T for x in encoded_reshaped)  # .shape = (n_batches, seq_length, batch_size)
        return batches
