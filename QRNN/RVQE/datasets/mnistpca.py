from ..data import *
from . import DataMNISTBase

import colorful
import pandas as pd
import math
import os.path as path


class DataMNISTPCABase(DataFactory):
    def __init__(self, shard: int, digits: List[int], resolution: int, pcs: int, **kwargs):
        super().__init__(shard, **kwargs)
        self.overrides_batch_size = True

        assert all(d in range(10) for d in digits), "digits have to be between 0 and 9"
        assert len(set(digits)) == len(digits), "duplicate digits"
        self.digits = digits

        # a batch of size 10 means 10 examples for each digit here
        self._old_batch_size = self.batch_size
        self.batch_size *= len(digits)

        assert 1 <= resolution <= 8, "resolution can be at most 8 bit"
        assert 1 <= pcs <= 8, "only 8 principal components precomputed"

        BIT_LUT = {n: tensor(int_to_bitword(n, 8))[:resolution].tolist() for n in range(2 ** 8)}

        # labels
        self.PC_RESOLUTION = resolution
        self.LABELS = [
            pad_and_chunk(
                int_to_bitword(i, width=math.ceil(math.log2(len(digits)))), self.PC_RESOLUTION
            )
            for i in range(len(digits))
        ]
        self.LABEL_LENGTH = len(self.LABELS[0])  # over how many measurements is the label spread
        self.PCs = pcs
        self.SEQ_LEN = self.LABEL_LENGTH - 1 + self.PCs

        # import as list of lists of bitwords, pad with label_length-1 zeros
        self._data = {
            stage: {
                digit: [
                    [BIT_LUT[val.item()] for val in row[:pcs]]
                    + [BIT_LUT[0]] * (self.LABEL_LENGTH - 1)
                    for row in DataMNISTBase._import_csv(
                        f"res/mnist-pca-8bit-{digit}-{stage_fn}.csv.gz"
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

        # we start with the label as soon as the last input has been presented
        self.TARGETS = [
            torch.cat((torch.zeros(self.PCs - 1, self.PC_RESOLUTION), torch.tensor(label).float(),))
            .int()
            .tolist()
            for label in self.LABELS
        ]

    @property
    def _batches(self) -> List[Batch]:
        raise NotImplementedError("next_batch overridden")

    @property
    def input_width(self) -> int:
        return self.PC_RESOLUTION

    def next_batch(self, step: int, stage: TrainingStage) -> Batch:
        # extract random batch of sentences
        sentences = []
        targets = []

        # get samples
        for _ in range(self._old_batch_size):
            for digit_idx, digit in enumerate(self.digits):
                data = self._data[stage][digit]
                data_idx = torch.randint(0, len(data), (1,), generator=self.rng).item()
                sentences.append(data[data_idx])
                targets.append(self.TARGETS[digit_idx])

        # turn into batch
        return self._sentences_to_batch(sentences, targets)

    def to_human(self, target: torch.LongTensor, offset: int = 0) -> str:
        if offset == 0 and not target.tolist() in self.TARGETS:
            return " ".join(bitword_to_str(t) for t in target)
        else:
            label = target[-self.LABEL_LENGTH :].tolist()
            if label in self.LABELS:
                return colorful.bold(str(self.digits[self.LABELS.index(label)]))
            else:
                return colorful.bold("?")

    def filter(self, sequence: torch.LongTensor, *, dim_sequence: int, **__) -> torch.LongTensor:
        """
            we expect these to be offset by 1 from a proper output of length 100, i.e. only of length 99
            we only care about the last self.LABEL_LENGTH pixels
        """
        assert sequence.dim() == 3 and dim_sequence in [1, 2]

        if dim_sequence == 1:
            return sequence[:, -self.LABEL_LENGTH :, :]
        elif dim_sequence == 2:
            return sequence[:, :, -self.LABEL_LENGTH :]

    def _ignore_output_at_step(self, index: int, target: Union[tensor, Bitword]) -> bool:
        """
            again we expect an input of length 99, so e.g. if the label has length 2,
            index 97 and 98 are the only ones not ignored
        """
        return index not in range(self.SEQ_LEN - 1 - self.LABEL_LENGTH, self.SEQ_LEN - 1)


# 2 bit resolution


class DataMNISTPCA_r2p8(DataMNISTPCABase):
    def __init__(self, shard: int, **kwargs):
        super().__init__(
            shard, digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], resolution=2, pcs=8, **kwargs
        )


class DataMNISTPCA_r2p5(DataMNISTPCABase):
    def __init__(self, shard: int, **kwargs):
        super().__init__(
            shard, digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], resolution=2, pcs=5, **kwargs
        )


class DataMNISTPCA_r2p2(DataMNISTPCABase):
    def __init__(self, shard: int, **kwargs):
        super().__init__(
            shard, digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], resolution=2, pcs=2, **kwargs
        )


# 3 bit resolution


class DataMNISTPCA_r3p8(DataMNISTPCABase):
    def __init__(self, shard: int, **kwargs):
        super().__init__(
            shard, digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], resolution=3, pcs=8, **kwargs
        )


class DataMNISTPCA_r3p5(DataMNISTPCABase):
    def __init__(self, shard: int, **kwargs):
        super().__init__(
            shard, digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], resolution=3, pcs=5, **kwargs
        )


class DataMNISTPCA_r3p2(DataMNISTPCABase):
    def __init__(self, shard: int, **kwargs):
        super().__init__(
            shard, digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], resolution=3, pcs=2, **kwargs
        )


# 4 bit resolution


class DataMNISTPCA_r4p8(DataMNISTPCABase):
    def __init__(self, shard: int, **kwargs):
        super().__init__(
            shard, digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], resolution=4, pcs=8, **kwargs
        )


class DataMNISTPCA_r4p5(DataMNISTPCABase):
    def __init__(self, shard: int, **kwargs):
        super().__init__(
            shard, digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], resolution=4, pcs=5, **kwargs
        )


class DataMNISTPCA_r4p2(DataMNISTPCABase):
    def __init__(self, shard: int, **kwargs):
        super().__init__(
            shard, digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], resolution=4, pcs=2, **kwargs
        )
