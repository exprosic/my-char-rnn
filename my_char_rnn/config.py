from collections import namedtuple


ModelParameter = namedtuple(
    'ModelParameters',
    [
        'batch_size',  # number of sequences in each batch
        'seq_length',  # length of sequence
        'n_layers',  # number of layers in the stacked LSTM cell
        'rnn_size',  # size of cell state in a single layer LSTM cell
        'grad_clip',  # magnitude for gradient clipping
        'learning_rate',
        'decay_rate',
    ]
)


# This recommended parameter is supposed to be copied rather than be imported.
_recommended_parameter = ModelParameter(
    batch_size=50,
    seq_length=50,
    n_layers=2,
    rnn_size=128,
    grad_clip=5.0,
    learning_rate=0.002,
    decay_rate=0.97,
)
