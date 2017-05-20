from collections import namedtuple


ModelParameter = namedtuple(
    'ModelParameters',
    [
        'data_name',
        'input_file_urls',
        'resource_path',
        'saved_session_file_name',

        'batch_size',  # number of sequences in each batch
        'seq_length',  # length of sequence
        'n_layers',  # number of layers in the stacked LSTM cell
        'rnn_size',  # size of cell state in a single layer LSTM cell
        'grad_clip',  # magnitude for gradient clipping
        'initial_learning_rate',
        'decay_rate',

        'schedule_decay_rate',  # decay rate for schedule sampling
    ]
)


# This recommended parameter is supposed to be copied rather than be imported.
_recommended_parameter = ModelParameter(
    data_name = 'shakespeare_5m',
    input_file_urls = {
        'shakespeare_5m': 'https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt',
    },
    resource_path = 'resource',
    saved_session_file_name='saved/model.ckpt',

    batch_size=50,
    seq_length=50,
    n_layers=2,
    rnn_size=128,
    grad_clip=5.0,
    initial_learning_rate=0.002,
    decay_rate=0.97,

    schedule_decay_rate=0.9,
)
