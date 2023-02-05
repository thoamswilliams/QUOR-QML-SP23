from typing import Tuple, List, Dict, Optional, Union, NewType
import torch
from torch import tensor

import itertools
import math


Bitword = NewType("Bitword", List[int])  # e.g. [0, 1, 1]
Batch = NewType("Batch", Tuple[torch.LongTensor, torch.LongTensor])


def zip_with_offset(a: torch.Tensor, b: torch.Tensor, offset: int = 1):
    """
        s -> (a0,b1), (a1,b2), (a2, b3), ...
    """
    return zip(a, b[offset:])


def pad_and_chunk(a: Bitword, size: int) -> List[Bitword]:
    """
        [1, 1, 0, 1, 1], 4 -> [[0, 0, 0, 1], [1, 0, 1, 1]]
    """
    a = [0] * (math.ceil(len(a) / size) * size - len(a)) + a
    return [a[i : i + size] for i in range(0, len(a), size)]


def bitword_to_int(lst: Union[Bitword, torch.LongTensor]) -> int:
    if not isinstance(lst, list):
        lst = lst.tolist()
    return int("".join(str(n) for n in lst), 2)


def bitword_to_onehot(lst: Union[Bitword, torch.LongTensor], width: int) -> torch.LongTensor:
    ret = torch.zeros(2 ** width)
    idx = bitword_to_int(lst)
    ret[idx] = 1.0
    return ret


def int_to_bitword_str(label: int, width: int) -> str:
    return bin(label)[2:].rjust(width, "0")


def int_to_bitword(label: int, width: Optional[int] = None) -> Bitword:
    if width is None:
        width = math.ceil(math.log2(label)) if label != 0 else 1
    return [int(c) for c in int_to_bitword_str(label, width)]


def bitword_to_str(lst: Union[Bitword, torch.LongTensor]) -> str:
    return int_to_bitword_str(bitword_to_int(lst), len(lst))


def char_to_bitword(char: str, characters: str, width: int) -> Bitword:
    idx = characters.index(char)
    char_bitword = f"{idx:b}".rjust(width, "0")
    return [int(c) for c in char_bitword[-width:]]


def bitword_to_char(bw: Bitword, characters: str) -> Bitword:
    return characters[bitword_to_int(bw)]


# character error rate
def character_error_rate(sequence: torch.LongTensor, target: torch.LongTensor) -> float:
    """
        we assume that sequence and target align 1:1
    """
    assert target.dim() == sequence.dim()
    return 1.0 - (sequence == target).all(dim=-1).to(float).mean()


# target preprocessing helper functions
def targets_for_loss(sentences: torch.LongTensor):
    """
        batch is B x L x W or just L x W
        B - batch
        L - sentence length
        W - word width
    """
    if sentences.dim() == 2:
        sentences = sentences.unsqueeze(0)
    assert sentences.dim() == 3

    return tensor([[bitword_to_int(word) for word in sentence] for sentence in sentences])


def skip_first(targets: torch.LongTensor) -> torch.LongTensor:
    """
        we never measure the first one, so skip that
        we assume B x L x W
        B - batch
        L - sentence length
        W - word width
    """
    return targets[:, 1:]


# data loader for distributed environment
from abc import ABC, abstractmethod
import enum


class TrainingStage(enum.Enum):
    TRAIN = 1
    VALIDATE = 2
    TEST = 3


class DataFactory(ABC):
    def __init__(
        self, shard: int, num_shards: int, batch_size: int, sentence_length: int, **kwargs
    ):
        self.shard = shard
        self.num_shards = num_shards
        self.batch_size = batch_size
        self.sentence_length = sentence_length
        # initial offset dependent on shard
        self._index = self.shard
        # local rng; derives seed from seed set in environment
        self.rng = torch.Generator().manual_seed(torch.randint(10 ** 10, (1,)).item() + shard)
        # by default, the dataset does not override the batch size
        self.overrides_batch_size = False

    def next_batch(self, step: int, stage: TrainingStage) -> Batch:
        # extract batch and advance pointer
        batch = self._batches[self._index]
        self._index += self.num_shards
        self._index %= len(self._batches)

        return batch

    def _sentences_to_batches(
        self, sentences: List[List[Bitword]], targets: List[List[Bitword]]
    ) -> List[Batch]:
        targets = tensor(targets)
        sentences = tensor(sentences)

        # split into batch-sized chunks
        targets = torch.split(targets, self.batch_size)
        sentences = torch.split(sentences, self.batch_size)

        return list(zip(sentences, targets))

    def _sentences_to_batch(
        self, sentences: List[List[Bitword]], targets: List[List[Bitword]]
    ) -> Batch:
        return self._sentences_to_batches(sentences, targets)[0]

    @property
    @abstractmethod
    def _batches(self) -> List[Batch]:
        pass

    @property
    @abstractmethod
    def input_width(self) -> int:
        pass

    @abstractmethod
    def to_human(self, target: torch.LongTensor, offset: int) -> str:
        """
            translate sentence to a nice human-readable form
            indenting by offset characters
        """
        pass

    def filter(
        self,
        sequence: torch.LongTensor,
        *,
        dim_sequence: int,
        targets_hint: torch.LongTensor,
        dim_targets: int,
    ) -> torch.LongTensor:
        """
            return a filtered sequence for use in validation loss and character error rate.
            The target tensor can be used as hint on what to filter.
        """
        return sequence

    def _ignore_output_at_step(self, index: int, target: Union[torch.LongTensor, Bitword]) -> bool:
        return False

    def ignore_output_at_step(self, index: int, targets: torch.LongTensor) -> bool:
        """
            return True if the output at this step is not expected
            to be a specific target; which means we can postselect it (using OAA)
            targets can be a batch or Bitword; we expect subclasses to override the _ method
            and simply stack the result
        """
        return tensor([self._ignore_output_at_step(index, target) for target in targets])
