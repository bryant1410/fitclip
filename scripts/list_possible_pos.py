#!/usr/bin/env python
import fileinput

from nltk.corpus import wordnet as wn
from nltk.corpus.reader import POS_LIST

if __name__ == "__main__":
    for line in fileinput.input():
        if line := line.strip():
            print("".join(pos for pos in POS_LIST if wn.synsets(line, pos=pos)))
