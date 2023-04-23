from ..data import *
import pandas as pd
import string


class TwitterSentiment(DataFactory):
    _data = None
    _targets = None
    VALID_CHARACTERS = "abcdefghijklmnopqrstuvwxyz,.?! \n"
    DISPLAY_CHARACTERS = "abcdefghijklmnopqrstuvwxyz,.?! ¶"
    assert len(VALID_CHARACTERS) <= 32, "characters should fit into 5 bits"

    @staticmethod
    def _load_shakespeare():
        import os.path as path

        DATASET_PATH = path.join(path.dirname(path.abspath(__file__)), "res/financetweets.csv")

        TwitterSentiment._data = []

        df = pd.read_csv(DATASET_PATH, delimiter = ';')
        input_text = df["text"].tolist()
        labels = df["label"].tolist()

        TwitterSentiment._targets = []

        for i in labels:
            for j in range(240):
                TwitterSentiment._targets.append(i)

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

            for i in range(240):
                if i < len(line):
                    c = line[i]
                else:
                    c = " "
                if not c in TwitterSentiment.VALID_CHARACTERS:
                    c = " "

                TwitterSentiment._data.append(
                    char_to_bitword(c, TwitterSentiment.VALID_CHARACTERS, 5)
                )
        print(len(TwitterSentiment._targets))
        print(len(TwitterSentiment._data))

    def __init__(self, shard: int, **kwargs):
        super().__init__(shard, **kwargs)
        self.sentence_length = 240

        if self._data == None:
            self._load_shakespeare()

    @property
    def _batches(self) -> List[Batch]:
        raise NotImplementedError("next_batch overridden")

    @property
    def input_width(self) -> int:
        return 5

    def next_batch(self, _, __) -> Batch:
        # extract random batch of sentences
        sentences = []
        targets = []
        while len(sentences) < self.batch_size:
            idx_start = torch.randint(
                0, int(len(self._data) / self.sentence_length), (1,), generator=self.rng
            ).item()
            sentences.append(self._data[idx_start * self.sentence_length : (idx_start + 1) * self.sentence_length])
            targets.append(self._targets[idx_start * self.sentence_length : (idx_start + 1) * self.sentence_length])
        print(len(sentences))
        print(len(targets))

        # turn into batch
        return self._sentences_to_batch(sentences, targets=targets)

    def to_human(self, target: torch.LongTensor, offset: int = 0) -> str:
        return " " * offset + "".join(
            [TwitterSentiment.DISPLAY_CHARACTERS[bitword_to_int(c)] for c in target]
        )
