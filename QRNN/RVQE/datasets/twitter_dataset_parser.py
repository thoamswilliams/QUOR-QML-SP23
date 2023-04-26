from ..data import *
import pandas as pd
import string


_data = None
_targets = None
VALID_CHARACTERS = "abcdefghijklmnopqrstuvwxyz? \n012"
DISPLAY_CHARACTERS = "abcdefghijklmnopqrstuvwxyz? P012"
assert len(VALID_CHARACTERS) <= 32, "characters should fit into 5 bits"

def load():
    import os.path as path

    DATASET_PATH = path.join(path.dirname(path.abspath(__file__)), "res/financetweets.csv")

    _data = []

    df = pd.read_csv(DATASET_PATH, delimiter = ';')
    input_text = df["text"].tolist()
    labels = df["label"].tolist()

    _targets = labels

    for i in labels:
        for j in range(240):
            _targets.append(char_to_bitword(i, VALID_CHARACTERS, 5))

    def onlyascii(char):
        if not char in string.printable:
            return ""
        else:
            return char

    def stringascii(string):
        output_string = ""

        for i in string:
            output_string += onlyascii(i)

        return output_string

    input_text = [stringascii(str(input_text[i])) for i in range(len(input_text))]
    for line in input_text:

        # cleanup
        line = line.rstrip().lower().replace("\r", "\n")
        for c, r in [
            ("'", ","),
            (";", ","),
            ('"', ","),
            (":", ","),
            ("1", "one"),
            ("2", "two"),
            ("3", "three"),
            ("4", "four"),
            ("5", "five"),
            ("6", "six"),
            ("7", "seven"),
            ("8", "eight"),
            ("9", "nine"),
            ("0", "zero"),
        ]:
            line = line.replace(c, r)
    

    


