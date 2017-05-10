import sys

from my_char_rnn.runner import train, generate


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'generate':
        generate()
    else:
        train()


if __name__ == '__main__':
    main()
