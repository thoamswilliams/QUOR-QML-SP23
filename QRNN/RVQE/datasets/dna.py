"""
    DNA Sequences
"""

import colorful

from ..data import *


class DataDNA(DataFactory):
    """
        we map GATC to 2-bit-strings; and U to the value 4.
        The task is to identify the DNA base following the single "C"
        which appears somewhere within the sequence
    """

    def __init__(self, shard: int, **kwargs):
        super().__init__(shard, **kwargs)

    @property
    def _batches(self) -> List[Batch]:
        raise NotImplementedError("next_batch overridden")

    @property
    def input_width(self) -> int:
        return 3

    def next_batch(self, _, __) -> Batch:
        # extract random batch of xor sequences like 011 101 110 000 ...
        sentences = []
        targets = []
        while len(sentences) < self.batch_size:
            sentence = torch.randint(0, 4, (self.sentence_length,), generator=self.rng)
            target = torch.zeros((self.sentence_length,)).int()

            # position of "C" anywhere but at the first and last index
            idx_U = torch.randint(1, self.sentence_length // 2, (1,), generator=self.rng).item()
            target[idx_U:] = sentence[idx_U]
            target[idx_U] = sentence[idx_U] = sentence[idx_U] + 4  # mark U

            # binarize
            sentence = [int_to_bitword(s, 3) for s in sentence.tolist()]
            target = [int_to_bitword(t, 3) for t in target.tolist()]

            sentences.append(sentence)
            targets.append(target)

        # turn into batch
        return self._sentences_to_batch(sentences, targets)

    def to_human(self, target: torch.LongTensor, offset: int = 0) -> str:
        target = tensor([bitword_to_int(t) for t in target])

        if offset == 0:  # gold
            is_input = max(target) >= 4
            idx_U = (target >= 4).nonzero()[0, 0].item()
            base_following_U = "GATC"[target[idx_U if is_input else -1].item() % 4]

            return f"base sequence U{colorful.bold(base_following_U)}  @  ago {self.sentence_length - idx_U}"

        elif offset == 1:  # comparison
            base_following_U = "GATCUUUU"[target[-1].item()]

            return f"base sequence U{colorful.bold(base_following_U)}"

    def filter(self, sequence: torch.LongTensor, *, dim_sequence: int, **__) -> torch.LongTensor:
        """
            we only care about the last item
        """
        assert sequence.dim() == 3 and dim_sequence in [1, 2]

        if dim_sequence == 1:
            return sequence[:, -1:, :]
        elif dim_sequence == 2:
            return sequence[:, :, -1:]

        # restore old shape
        return sequence_out if dim_sequence == 1 else sequence_out.transpose(1, 2)

    def _ignore_output_at_step(self, index: int, target: Union[tensor, Bitword]) -> bool:
        """
            we expect target to have length self.sequence_length - 1, so the last index
            is self.sequence_length - 2
        """
        return index != self.sentence_length - 2
