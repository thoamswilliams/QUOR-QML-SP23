from ..data import *


class DataShakespeare(DataFactory):
    _data = None
    VALID_CHARACTERS = "abcdefghijklmnopqrstuvwxyz,.?! \n"
    DISPLAY_CHARACTERS = "abcdefghijklmnopqrstuvwxyz,.?! ¶"
    assert len(VALID_CHARACTERS) <= 32, "characters should fit into 5 bits"

    @staticmethod
    def _load_shakespeare():
        import os.path as path

        SHAKESPEARE_PATH = path.join(path.dirname(path.abspath(__file__)), "res/shakespeare.txt")

        DataShakespeare._data = []

        with open(SHAKESPEARE_PATH, "r") as f:
            leftover = set()

            for i, line in enumerate(f):
                if i < 245 or i > 124440:  # strip license info
                    continue

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

                for c in line:
                    if not c in DataShakespeare.VALID_CHARACTERS:
                        c = " "
                    DataShakespeare._data.append(
                        char_to_bitword(c, DataShakespeare.VALID_CHARACTERS, 5)
                    )

    def __init__(self, shard: int, **kwargs):
        super().__init__(shard, **kwargs)

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
        while len(sentences) < self.batch_size:
            idx_start = torch.randint(
                0, len(self._data) - self.sentence_length, (1,), generator=self.rng
            ).item()
            sentences.append(self._data[idx_start : idx_start + self.sentence_length])

        # turn into batch
        return self._sentences_to_batch(sentences, targets=sentences)

    def to_human(self, target: torch.LongTensor, offset: int = 0) -> str:
        return " " * offset + "".join(
            [DataShakespeare.DISPLAY_CHARACTERS[bitword_to_int(c)] for c in target]
        )
