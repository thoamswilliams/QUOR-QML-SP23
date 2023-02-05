from ..data import *

import colorful
import pandas as pd
import math
import os.path as path


class DataMNISTBase(DataFactory):
    @staticmethod
    def _import_csv(filename: str) -> torch.LongTensor:
        return tensor(
            pd.read_csv(
                path.join(path.dirname(path.abspath(__file__)), filename,),
                header=None,
                compression="gzip",
            ).values
        )

    def __init__(
        self,
        shard: int,
        digits: List[int],
        scanlines: List[int],
        deskewed: bool,
        large: bool,
        **kwargs,
    ):
        super().__init__(shard, **kwargs)
        self.overrides_batch_size = True

        assert all(d in range(10) for d in digits), "digits have to be between 0 and 9"
        assert len(set(digits)) == len(digits), "duplicate digits"
        self.digits = digits

        # a batch of size 10 means 10 examples for each digit here
        self._old_batch_size = self.batch_size
        self.batch_size *= len(digits)

        assert all(s in range(3) for s in scanlines), "scanlines are 0, 1, or 2"
        assert len(set(scanlines)) == len(scanlines), "duplicate scanlines"
        self.scanlines = scanlines

        BIT_LUT = {n: tensor(int_to_bitword(n, 3))[scanlines].tolist() for n in range(8)}

        SUBSET = ""
        if deskewed:
            SUBSET += "-deskewed"
        if large:
            SUBSET += "-large"

        # import as list of lists of bitwords
        self._data = {
            stage: {
                digit: [
                    [BIT_LUT[val.item()] for val in row]
                    for row in DataMNISTBase._import_csv(
                        f"res/mnist-simple-{digit}{SUBSET}-{stage_fn}.csv.gz"
                    )
                ]
                for digit in digits
            }
            for stage, stage_fn in [
                [TrainingStage.TRAIN, "train"],
                [TrainingStage.VALIDATE, "validate"],
                [TrainingStage.TEST, "test"],
            ]
        }

        # import centroids
        self._data_centroids = {
            digit: [BIT_LUT[val.item()] for val in row]
            for digit, row in enumerate(
                DataMNISTBase._import_csv(f"res/mnist-centroids{SUBSET}.csv.gz")
            )
        }

        self.SEQ_LEN = len(self._data_centroids[0])  # sequence length
        self.IMAGE_SIZE = round(math.sqrt(self.SEQ_LEN))  # e.g. 10x10, or 20x20

        self.num_scanlines = len(scanlines)  # number of in/out qubits
        self.LABELS = [
            pad_and_chunk(
                int_to_bitword(i, width=math.ceil(math.log2(len(digits)))), self.num_scanlines
            )
            for i in range(len(digits))
        ]
        self.label_length = len(self.LABELS[0])  # over how many measurements is the label spread

        # last pixels contain the label
        self.TARGETS = [
            torch.cat(
                (
                    torch.zeros(self.SEQ_LEN - self.label_length, self.num_scanlines),
                    torch.tensor(label).float(),
                )
            )
            .int()
            .tolist()
            for label in self.LABELS
        ]

    @property
    def _batches(self) -> List[Batch]:
        raise NotImplementedError("next_batch overridden")

    @property
    def input_width(self) -> int:
        return self.num_scanlines

    def next_batch(self, step: int, stage: TrainingStage) -> Batch:
        # extract random batch of sentences
        sentences = []
        targets = []

        # we switch back and forth between centroid and general samples
        ANNEALING_PERIOD = 100
        frac_centroids = 0  # math.cos(10 * step / (2 * math.pi) / ANNEALING_PERIOD) ** 4
        centroid_samples = round(self._old_batch_size * frac_centroids)
        normal_samples = self._old_batch_size - centroid_samples

        # get normal samples
        for _ in range(normal_samples):
            for digit_idx, digit in enumerate(self.digits):
                data = self._data[stage][digit]
                data_idx = torch.randint(0, len(data), (1,), generator=self.rng).item()
                sentences.append(data[data_idx])
                targets.append(self.TARGETS[digit_idx])

        # get centroid samples (yes, this adds the same sample multiple times)
        for _ in range(centroid_samples):
            for digit_idx, digit in enumerate(self.digits):
                sentences.append(self._data_centroids[digit])
                targets.append(self.TARGETS[digit_idx])

        # turn into batch
        return self._sentences_to_batch(sentences, targets)

    def _print_as_images(self, target: torch.LongTensor, offset: int = 0) -> str:
        # the predicted image is missing the first pixel
        if len(target) == self.SEQ_LEN - 1:
            target = torch.cat((torch.zeros(1, self.num_scanlines, dtype=int), target))

        # split scanlines
        scanlines = target.transpose(0, 1)

        # reshape to image
        images = [t.reshape(self.IMAGE_SIZE, -1) for t in scanlines]

        # group two rows per image
        images = [[t.transpose(0, 1) for t in img.split(2)] for img in images]

        # print
        PIXEL_REP = " ▄▀█"
        out = ""

        for lines in zip(*images):
            out += "\t" + "".join(bitword_to_char(d, PIXEL_REP) for d in lines[0])
            for line in lines[1:]:
                out += "  " + "".join(bitword_to_char(d, PIXEL_REP) for d in line)
            out += "\n"

        return out[:-1]

    def to_human(self, target: torch.LongTensor, offset: int = 0) -> str:
        if offset == 0 and not target.tolist() in self.TARGETS:
            return self._print_as_images(target)
        else:
            label = target[-self.label_length :].tolist()
            if label in self.LABELS:
                return colorful.bold(str(self.digits[self.LABELS.index(label)]))
            else:
                return colorful.bold("?")

    def filter(self, sequence: torch.LongTensor, *, dim_sequence: int, **__) -> torch.LongTensor:
        """
            we expect these to be offset by 1 from a proper output of length 100, i.e. only of length 99
            we only care about the last self.label_length pixels
        """
        assert sequence.dim() == 3 and dim_sequence in [1, 2]

        if dim_sequence == 1:
            return sequence[:, -self.label_length :, :]
        elif dim_sequence == 2:
            return sequence[:, :, -self.label_length :]

    def _ignore_output_at_step(self, index: int, target: Union[tensor, Bitword]) -> bool:
        """
            again we expect an input of length 99, so e.g. if the label has length 2,
            index 97 and 98 are the only ones not ignored
        """
        return index not in range(self.SEQ_LEN - 1 - self.label_length, self.SEQ_LEN - 1)


class DataMNIST01(DataMNISTBase):
    def __init__(self, shard: int, **kwargs):
        super().__init__(
            shard, digits=[0, 1], scanlines=[0, 1], deskewed=False, large=False, **kwargs
        )


class DataMNIST36(DataMNISTBase):
    def __init__(self, shard: int, **kwargs):
        super().__init__(
            shard, digits=[3, 6], scanlines=[0, 1], deskewed=False, large=False, **kwargs
        )


class DataMNIST8(DataMNISTBase):
    def __init__(self, shard: int, **kwargs):
        super().__init__(
            shard,
            digits=[0, 1, 2, 3, 4, 5, 6, 7, 8],
            scanlines=[0, 1, 2],
            deskewed=False,
            large=False,
            **kwargs,
        )


class DataMNIST01ds(DataMNISTBase):
    def __init__(self, shard: int, **kwargs):
        super().__init__(
            shard, digits=[0, 1], scanlines=[0, 1], deskewed=True, large=False, **kwargs
        )


class DataMNIST36ds(DataMNISTBase):
    def __init__(self, shard: int, **kwargs):
        super().__init__(
            shard, digits=[3, 6], scanlines=[0, 1], deskewed=True, large=False, **kwargs
        )


class DataMNIST8ds(DataMNISTBase):
    def __init__(self, shard: int, **kwargs):
        super().__init__(
            shard,
            digits=[0, 1, 2, 3, 4, 5, 6, 7],
            scanlines=[0, 1, 2],
            deskewed=True,
            large=False,
            **kwargs,
        )


class DataMNIST01ds_lrg(DataMNISTBase):
    def __init__(self, shard: int, **kwargs):
        super().__init__(
            shard, digits=[0, 1], scanlines=[0, 1], deskewed=True, large=True, **kwargs
        )


class DataMNIST36ds_lrg(DataMNISTBase):
    def __init__(self, shard: int, **kwargs):
        super().__init__(
            shard, digits=[3, 6], scanlines=[0, 1], deskewed=True, large=True, **kwargs
        )


class DataMNIST8ds_lrg(DataMNISTBase):
    def __init__(self, shard: int, **kwargs):
        super().__init__(
            shard,
            digits=[0, 1, 2, 3, 4, 5, 6, 7],
            scanlines=[0, 1, 2],
            deskewed=True,
            large=True,
            **kwargs,
        )


# EVEN


class DataMNIST_EvenOdd(DataMNISTBase):
    def __init__(self, shard: int, **kwargs):
        super().__init__(
            shard,
            digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            scanlines=[0, 1],
            deskewed=False,
            large=False,
            **kwargs,
        )
        self.LABELS = [[int_to_bitword(d % 2, width=self.num_scanlines)] for d in self.digits]
        self.label_length = 1  # even and odd are just one bit

        # last pixels contain the label
        self.TARGETS = [
            torch.cat(
                (
                    torch.zeros(self.SEQ_LEN - self.label_length, self.num_scanlines),
                    torch.tensor(label).float(),
                )
            )
            .int()
            .tolist()
            for label in self.LABELS
        ]

    def to_human(self, target: torch.LongTensor, offset: int = 0) -> str:
        if offset == 0 and not target.tolist() in self.TARGETS:
            return self._print_as_images(target)
        else:
            label = target[-self.label_length :].tolist()
            if label in self.LABELS:
                return colorful.bold(["even", "odd"][self.LABELS.index(label)])
            else:
                return colorful.bold("?")


# GENERATIVE


class DataMNIST01_Gen(DataMNISTBase):
    """
        generate 10 x 10 binary images of 0s and 1s
        The first pixel is the label.
        As the first pixel is ignored in the prediction anyhow, there's no filtering.
    """

    def __init__(self, shard: int, **kwargs):
        super().__init__(
            shard, digits=[0, 1], scanlines=[0, 1], deskewed=True, large=False, **kwargs
        )

    def next_batch(self, _, stage: TrainingStage) -> Batch:
        # extract random batch of sentences
        sentences = []

        # we try to keep it balanced between 0 and 1, even if batch size is 1, and multiple shards are used
        dataA = self._data[stage][0]
        dataB = self._data[stage][1]
        labelA = [0, 0]
        labelB = [0, 1]
        if self.shard % 2 == 1:
            dataA, dataB = dataB, dataA
            labelA, labelB = labelB, labelA

        while len(sentences) < self.batch_size // 2:
            idx = torch.randint(0, len(dataA), (1,), generator=self.rng).item()
            sentence = dataA[idx].copy()
            sentence[0] = labelA
            sentences.append(sentence)

        while len(sentences) < self.batch_size:
            idx = torch.randint(0, len(dataB), (1,), generator=self.rng).item()
            sentence = dataB[idx].copy()
            sentence[0] = labelB
            sentences.append(sentence)

        # turn into batch
        return self._sentences_to_batch(sentences, targets=sentences)

    def to_human(self, target: torch.LongTensor, offset: int = 0) -> str:
        return super()._print_as_images(target)

    def filter(self, sequence: torch.LongTensor, *, dim_sequence: int, **__) -> torch.LongTensor:
        return sequence

    def _ignore_output_at_step(self, index: int, target: Union[tensor, Bitword]) -> bool:
        return False
