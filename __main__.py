import sys
from mutable import Settingsiterator

def main(argv):
    config_file = argv[0]
    print(config_file)
    Settingsiterator.iterate(config_file)

if __name__ == "__main__":
    main(sys.argv[1:])