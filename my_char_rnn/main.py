from __future__ import print_function

import argparse
import sys

from config import _recommended_parameter as param
from my_char_rnn.runner import train, generate


def main():
    parser = argparse.ArgumentParser(description='Train a model or sample from an existing model.')
    parser.add_argument('action', help='"train" or "generate"')
    parser.add_argument('--save_sample', metavar='FILE_NAME', nargs='?', const=param.saved_sample_file_name, help='save the generated sample to file FILE_NAME')
    parser.add_argument('--save_states', action='store_true', help='whether to save the hidden states')
    args = parser.parse_args()

    if args.action=='generate':
        generate(file_name=args.save_sample, save_states=args.save_states)
    elif args.action=='train':
        train()
    else:
        print('unknown action "{}"'.format(args.action), file=sys.stderr)


if __name__ == '__main__':
    main()
