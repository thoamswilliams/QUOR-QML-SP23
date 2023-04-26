from ..data import *
import pandas as pd
import string
import os.path as path


class AmazonSentiment(DataFactory):
    _train_data = None
    _train_targets = None
    _test_data = None
    _test_targets = None
    VALID_CHARACTERS = "abcdefghijklmnopqrstuvwxyz? \n012"
    DISPLAY_CHARACTERS = "abcdefghijklmnopqrstuvwxyz? P012"
    assert len(VALID_CHARACTERS) <= 32, "characters should fit into 5 bits"

    @staticmethod
    def _load_shakespeare():
        DATASET_PATH_TRAIN = path.join(path.dirname(path.abspath(__file__)), "res/amzn_train1.csv")

        AmazonSentiment._train_data = []

        df = pd.read_csv(DATASET_PATH_TRAIN, delimiter = ',')
        input_text = df["sentences"].tolist()
        labels = df["labels"].tolist()

        AmazonSentiment._train_targets = []

        for i in labels:
            if (i == "[1, 0]"):
                char = 1
            elif (i == "[0, 1]"):
                char = 0
            for j in range(80):
                AmazonSentiment._train_targets.append(char_to_bitword(str(char), AmazonSentiment.VALID_CHARACTERS, 5))

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

            for i in range(80):
                if i < len(line):
                    c = line[i]
                else:
                    c = " "
                if not c in AmazonSentiment.VALID_CHARACTERS:
                    c = " "

                AmazonSentiment._train_data.append(
                    char_to_bitword(c, AmazonSentiment.VALID_CHARACTERS, 5)
                )
        #------------------ inport test data
        DATASET_PATH_TEST = path.join(path.dirname(path.abspath(__file__)), "res/amzn_test1.csv")

        AmazonSentiment._test_data = []

        df = pd.read_csv(DATASET_PATH_TEST, delimiter = ';')
        input_text = df["text"].tolist()
        labels = df["label"].tolist()

        AmazonSentiment._test_targets = []

        for i in labels:
            if (i == "[1, 0]"):
                char = 1
            elif (i == "[0, 1]"):
                char = 0
            for j in range(80):
                AmazonSentiment._test_targets.append(char_to_bitword(char, AmazonSentiment.VALID_CHARACTERS, 5))

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

            for i in range(80):
                if i < len(line):
                    c = line[i]
                else:
                    c = " "
                if not c in AmazonSentiment.VALID_CHARACTERS:
                    c = " "

                AmazonSentiment._test_data.append(
                    char_to_bitword(c, AmazonSentiment.VALID_CHARACTERS, 5)
                )



    def __init__(self, shard: int, **kwargs):
        super().__init__(shard, **kwargs)
        self.sentence_length = 80

        if self._train_data == None:
            self._load_shakespeare()

    @property
    def _batches(self) -> List[Batch]:
        raise NotImplementedError("next_batch overridden")

    @property
    def input_width(self) -> int:
        return 5

    def next_batch(self, _, stage) -> Batch:
        # extract random batch of sentences
        sentences = []
        targets = []

        if (stage == TrainingStage.TRAIN):
            while len(sentences) < self.batch_size:
                idx_start = torch.randint(
                    0, int(len(self._train_data) / self.sentence_length), (1,), generator=self.rng
                ).item()
                sentences.append(self._train_data[idx_start * self.sentence_length : (idx_start + 1) * self.sentence_length])
                targets.append(self._train_targets[idx_start * self.sentence_length : (idx_start + 1) * self.sentence_length])
        elif (stage == TrainingStage.VALIDATE):
            while len(sentences) < self.batch_size:
                idx_start = torch.randint(
                    0, int(len(self._test_data) / self.sentence_length), (1,), generator=self.rng
                ).item()
                sentences.append(self._test_data[idx_start * self.sentence_length : (idx_start + 1) * self.sentence_length])
                targets.append(self._test_targets[idx_start * self.sentence_length : (idx_start + 1) * self.sentence_length])

        # turn into batch
        return self._sentences_to_batch(sentences, targets=targets)

    def to_human(self, target: torch.LongTensor, offset: int = 0) -> str:
        return " " * offset + "".join(
            [AmazonSentiment.DISPLAY_CHARACTERS[bitword_to_int(c)] for c in target]
        )