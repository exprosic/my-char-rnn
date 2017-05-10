import numpy as np

from my_char_rnn.config import ModelParameter


class DataLoader(object):
    def __init__(self, param, file_name):
        assert isinstance(param, ModelParameter)
        assert isinstance(file_name, str)
        self.param = param
        self.file_name = file_name

        # Below are just placeholders for type-inferring, which will be assigned by self.pre_process
        self.vocab = self.r_vocab = self.n_batches = self.vocab_size = None
        self._full_text = None

    def pre_process(self):
        """
        Load the full text, calculate the vocabulary and update the parameters.
        """
        self._full_text = ''.join(open(self.file_name).readlines())  # full text in a str
        self.vocab = sorted(set(self._full_text))  # "abcdefg..."
        self.r_vocab = {c: i for i, c in enumerate(self.vocab)}  # {'a': 0, 'b': 1, ...}
        self.n_batches = len(self._full_text) // (self.param.batch_size * self.param.seq_length)
        self.vocab_size = len(self.vocab)

    def get_batches(self):
        """
        Batch generator.

        :return: One batch of (input, target) at each iteration.
        """
        full_input_text = self._full_text[:self.n_batches * self.param.batch_size * self.param.seq_length]
        full_output_text = full_input_text[1:] + full_input_text[:1]
        return zip(self._text_to_batches(full_input_text), self._text_to_batches(full_output_text))

    def _text_to_batches(self, text):
        """
        Translate each character into one-hot encoding vector and reshape.

        :param text: str
        :return: ndarray with shape (n_batches, seq_length, batch_size)
        """
        assert len(text) == self.n_batches * self.param.batch_size * self.param.seq_length
        encoded = np.array([self.r_vocab[c] for c in text])  # .dtype=int, .shape=(len(text),)
        encoded_reshaped = encoded.reshape((self.n_batches, self.param.batch_size, self.param.seq_length))
        batches = (x.T for x in encoded_reshaped)  # .shape = (n_batches, seq_length, batch_size)
        return batches
