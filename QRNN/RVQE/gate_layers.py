from typing import Union, List, Tuple

from .quantum import *
import torch
from torch import nn, tensor

import copy
import math

import itertools

_gate_layer_id = 0


class GateLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.dagger = False

        global _gate_layer_id
        _gate_layer_id += 1
        self._id = _gate_layer_id

    @property
    def Ut(self) -> Operator:
        U = self.U
        shape = U.shape
        dim = tensor(U.shape[: len(U.shape) // 2]).prod()  # product of half of the dimensions
        return U.reshape(dim, dim).T.reshape(shape)

    @property
    def T(self):
        new = copy.copy(self)
        new.dagger = not new.dagger
        return new

    def forward(self, kob: KetOrBatch, normalize_afterwards: bool = False) -> KetOrBatch:
        kob = apply(self.U if not self.dagger else self.Ut, kob, self.lanes)
        return kob if not normalize_afterwards else normalize(kob)

    def extra_repr(self):
        return f"lanes={self.lanes}, id={self._id}{', †' if self.dagger else ''}"

    def to_mat(self):
        n = num_operator_qubits(self.U)
        basis = list(itertools.product("01", repeat=n))

        out = tensor(
            [[dot(ket("".join(x)), self.forward(ket("".join(y)))) for x in basis] for y in basis]
        )

        return out


class XLayer(GateLayer):
    def __init__(self, target_lane: int):
        super().__init__()
        self.lanes = [target_lane]
        self.U = tensor([[0.0, 1.0], [1.0, 0.0]])


class HLayer(GateLayer):
    def __init__(self, target_lane: int):
        super().__init__()
        self.lanes = [target_lane]
        self.U = tensor([[1.0, 1.0], [1.0, -1.0]]) / math.sqrt(2.0)


class rYLayer(GateLayer):
    def __init__(self, target_lane: int, initial_θ: Union[float, tensor, nn.Parameter]):
        super().__init__()
        assert isinstance(initial_θ, (float, torch.Tensor, nn.Parameter)), "wrong angle type given"

        self.lanes = [target_lane]
        self.θ = initial_θ

        if isinstance(self.θ, float):
            # assume this is a constant rotation gate, so create cache value
            self.θ = tensor(self.θ)
            self._U = self.U

    @property
    def U(self) -> Operator:
        if hasattr(self, "_U"):
            return self._U

        # note: these matrices are TRANSPOSED! in this notation
        θ = self.θ
        return torch.stack(
            [
                torch.stack([(0.5 * θ).cos(), (-0.5 * θ).sin()]),
                torch.stack([(0.5 * θ).sin(), (0.5 * θ).cos()]),
            ]
        )


class crYLayer(GateLayer):
    def __init__(
        self,
        control_lane: Union[int, List[int], Tuple[int]],
        target_lane: int,
        initial_φ: float = 1.0,
    ):
        super().__init__()

        if isinstance(control_lane, int):
            self.lanes = [control_lane, target_lane]
        else:  # assume a list of control lanes given
            self.lanes = [*control_lane, target_lane]

        self.φ = nn.Parameter(tensor(initial_φ))

    @property
    def U(self) -> Operator:
        # note: these matrices are TRANSPOSED! in this notation
        φ = self.φ
        rY = torch.stack(
            [
                torch.stack([(0.5 * φ).cos(), (-0.5 * φ).sin()]),
                torch.stack([(0.5 * φ).sin(), (0.5 * φ).cos()]),
            ]
        )
        return ctrlMat(rY, len(self.lanes) - 1)


class cmiYLayer(GateLayer):
    def __init__(self, control_lane: int, target_lane: int):
        super().__init__()

        self.lanes = [control_lane, target_lane]

        # note: these matrices are TRANSPOSED! in this notation
        self.U = tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0, 0.0],
            ]
        ).T.reshape(2, 2, 2, 2)


class PostselectLayer(GateLayer):
    def __init__(self, target_lane: int, on: int):
        super().__init__()
        self.on = on
        self.lanes = [target_lane]
        self.U = torch.zeros(4).reshape(2, 2)
        self.U[on, on] = 1.0

    def forward(self, kob: KetOrBatch) -> KetOrBatch:
        return super().forward(kob, normalize_afterwards=True)

    def extra_repr(self):
        return super().extra_repr() + f", on={self.on}"
