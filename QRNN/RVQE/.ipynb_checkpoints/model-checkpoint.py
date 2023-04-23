from typing import Tuple, List, Dict, Optional, Union, Callable

from .compound_layers import (
    UnitaryLayer,
    QuantumNeuronLayer,
    FastQuantumNeuronLayer,
    BitFlipLayer,
    PostselectManyLayer,
)
from .quantum import *
from .data import zip_with_offset, int_to_bitword, bitword_to_int, Bitword

import torch
from torch import nn

import math


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def thread_inputs_over_batch(batch: KetBatch, op, inputs: torch.LongTensor) -> KetBatch:
    assert is_batch(batch) and inputs.dim() == 2, "cannot thread inputs over batch"
    out = []
    for i, inpt in enumerate(inputs):
        out.append(op(inpt).forward(batch[i]))
    return mark_batch(torch.stack(out))


class RVQECell(nn.Module):
    def __init__(
        self,
        *,
        workspace_size: int,
        input_size: int,
        stages: int,
        order: int = 2,
        degree: int = 2,
        bias: float = math.pi / 2,
        fast: bool = True,
    ):
        """
            by default we set up the qubit indices to be
            input, workspace, ancillas
        """
        assert (
            workspace_size >= 1 and input_size >= 1 and stages >= 1 and order >= 1
        ), "all parameters have to be >= 1"

        super().__init__()

        self.stages = stages
        self.order = order
        self.degree = degree
        self.bias = bias

        QNL_T = FastQuantumNeuronLayer if fast else QuantumNeuronLayer
        ancilla_count = QNL_T.ancillas_for_order(order)

        self.inout = list(range(0, input_size))
        self.workspace = list(range(input_size, input_size + workspace_size))
        self.ancillas = list(
            range(input_size + workspace_size, input_size + workspace_size + ancilla_count)
        )

        self.input_layer = nn.Sequential(
            *[
                QNL_T(
                    workspace=self.workspace + self.inout,
                    outlane=out,
                    order=order,
                    ancillas=self.ancillas,
                    degree=degree,
                    bias=bias,
                )
                for out in self.workspace
            ]
        )
        self.kernel_layer = nn.Sequential(
            *[
                nn.Sequential(
                    UnitaryLayer(self.workspace),
                    *[
                        QNL_T(
                            workspace=self.workspace + self.inout,
                            outlane=out,
                            order=order,
                            ancillas=self.ancillas,
                            degree=degree,
                            bias=bias,
                        )
                        for out in self.workspace
                    ],
                )
                for _ in range(stages)
            ]
        )
        self.output_layer = nn.Sequential(
            *[
                QNL_T(
                    workspace=self.workspace + self.inout,
                    outlane=out,
                    order=order,
                    ancillas=self.ancillas,
                    degree=degree,
                    bias=bias,
                )
                for out in self.inout
            ]
        )

    @property
    def num_qubits(self) -> int:
        return len(self.ancillas) + len(self.workspace) + len(self.inout)

    def forward(self, batch: KetBatch, inputs: torch.LongTensor) -> Tuple[KetBatch, KetBatch]:
        """
            we expect a batch of states and inputs
        """
        # we assume all psi's in kob has its input lanes reset to 0
        assert inputs.shape[-1] == len(self.inout), "wrong input size given"
        assert (
            num_state_qubits(batch) == self.num_qubits
        ), "state given does not have the right size"
        assert is_batch(batch) and inputs.dim() == 2, "expecting batched call"

        # set input
        batch = self.bitflip_batch(batch, inputs)

        # input and kernel layers don't write to the inout lanes, it is read only
        batch = self.input_layer.forward(batch)
        batch = self.kernel_layer.forward(batch)

        # reset input
        batch = self.bitflip_batch(batch, inputs)

        # write to output layer
        batch = self.output_layer.forward(batch)

        return probabilities(batch, self.inout), batch

    def bitflip_batch(self, batch: KetBatch, inputs: torch.LongTensor) -> KetBatch:
        # reset input one batch element at a time
        return thread_inputs_over_batch(batch, self._bitflip_for, inputs)

    def _bitflip_for(self, input: Bitword) -> BitFlipLayer:
        return BitFlipLayer([lane for i, lane in enumerate(self.inout) if input[i] == 1])


class RVQE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.cell = RVQECell(**kwargs)

    def forward(
        self,
        inputs: torch.LongTensor,
        targets: torch.LongTensor,
        postselect_measurement: Union[bool, Callable[[int], bool]],
    ) -> Tuple[KetOrBatch, KetOrBatch, float]:
        assert inputs.shape == targets.shape, "inputs and targets have to have same shape"
        assert inputs.dim() in [2, 3], "unbatched input has to have dim 2, batched dim 3"

        # prepare batch
        was_batched = inputs.dim() == 3

        if not was_batched:
            inputs = inputs.unsqueeze(0)
            targets = targets.unsqueeze(0)

        BATCH = batch_size(inputs)

        if isinstance(postselect_measurement, bool):
            # return callback that gives constant
            _postselect_measurement = postselect_measurement
            postselect_measurement = lambda _, trgt: tensor(_postselect_measurement).expand(BATCH)

        # now the shape of the input and target tensors is
        # BATCH x LEN x WIDTH
        # but we need
        # LEN x BATCH x WIDTH
        inputs = inputs.transpose(0, 1)
        targets = targets.transpose(0, 1)

        kob = ket_to_batch(ket0(self.cell.num_qubits), copies=BATCH, share_memory=False)

        probs = []
        measured_seq = []
        min_postsel_prob = tensor(1.0)

        for i, (inpt, trgt) in enumerate(zip_with_offset(inputs, targets)):
            p, kob = self.cell.forward(kob, inpt)
            probs.append(p)

            # measure or postselect output
            measure = []
            ps_mask = postselect_measurement(i, trgt)
            for sample_idx, ps_sample in enumerate(ps_mask):
                if ps_sample:
                    # we postselect this sample within the batch
                    measure.append(trgt[sample_idx])
                else:
                    # randomly measure
                    output_distribution = torch.distributions.Categorical(probs=p[sample_idx])
                    sample = output_distribution.sample()
                    measure.append(tensor(int_to_bitword(sample, width=len(self.cell.inout))))
            measure = torch.stack(measure)

            kob = thread_inputs_over_batch(
                kob, lambda m: PostselectManyLayer(self.cell.inout, m), measure
            )
            postsel_prob = tensor(
                [p[i, ii] for i, ii in enumerate(bitword_to_int(m) for m in measure)]
            )

            if any(ps_mask):
                min_postsel_prob = min(min_postsel_prob, postsel_prob[ps_mask].min())
            measured_seq.append(measure)

            # reset qubits
            kob = self.cell.bitflip_batch(kob, measure)

        probs = torch.stack(probs)
        measured_seq = torch.stack(measured_seq)

        # shape of probs and measured sequence is
        # LEN x BATCH x WIDTH
        # but want
        # BATCH x LEN x WIDTH for measured_seq, and
        # BATCH x WIDTH x LEN for probs, as for loss we need BATCH x CLASS x d

        measured_seq = measured_seq.transpose(0, 1)
        probs = probs.transpose(0, 1).transpose(1, 2)

        if was_batched:
            return probs, measured_seq, min_postsel_prob.item()
        else:
            return probs[0], measured_seq[0], min_postsel_prob.item()
