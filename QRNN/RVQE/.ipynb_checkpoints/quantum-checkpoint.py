from typing import Tuple, List, Dict, Optional, Union, NewType

import torch
from torch import tensor
from math import pi, sqrt

import functools


# QUANTUM STATES, BATCHES, AND OPERATORS

Ket = NewType("Ket", torch.Tensor)
KetBatch = NewType("KetBatch", torch.Tensor)
Operator = NewType("Operator", torch.Tensor)


def ket0(num_qubits: int) -> Ket:
    psi = torch.zeros(2 ** num_qubits)
    psi[0] = 1
    return psi.reshape(shape=[2] * num_qubits)


def ket1(num_qubits: int) -> Ket:
    psi = torch.zeros(2 ** num_qubits)
    psi[-1] = 1
    return psi.reshape(shape=[2] * num_qubits)


def ket(descr: str) -> Ket:
    out = None
    for s in descr:
        if s == "0":
            psi = tensor([1.0, 0.0])
        elif s == "1":
            psi = tensor([0.0, 1.0])
        elif s == "+":
            psi = normalize(tensor([1.0, 1.0]))
        elif s == "-":
            psi = normalize(tensor([1.0, -1.0]))
        else:
            assert False, "description has to be one of 0, 1, + or -"

        if out is None:
            out = psi
        else:
            out = torch.ger(out, psi).view(-1)

    return out.reshape(shape=[2] * len(descr))


# BATCHES
# we store an attribute in the tensor called _is_batch as we cannot properly subclass torch.Tensor

KetOrBatch = NewType("KetOrBatch", Union[Ket, KetBatch])


def mark_batch(batch: KetBatch) -> KetBatch:
    batch._is_batch = True
    return batch


def ket_to_batch(psi: Ket, copies: int, share_memory: bool = True) -> KetBatch:
    batch = (
        psi.expand(copies, *psi.shape)
        if share_memory
        else psi.repeat(copies, *[1 for _ in psi.shape])
    )
    return mark_batch(batch)


def is_batch(something) -> bool:
    return isinstance(something, torch.Tensor) and hasattr(something, "_is_batch")


def batch_size(batch: KetBatch) -> int:
    return batch.shape[0]


def mark_batch_like(model: KetOrBatch, to_mark: KetOrBatch) -> KetOrBatch:
    assert isinstance(to_mark, torch.Tensor) and isinstance(
        model, torch.Tensor
    ), "both arguments have to be torch.Tensor"
    assert not (is_batch(to_mark) and not is_batch(model)), "cannot unmark batch"
    return mark_batch(to_mark) if is_batch(model) else to_mark


# EINSUM INDICES

# indices available to einsum. We reserve "a" for the batch dimension
_EINSUM_BATCH_CHARACTER = "a"
_EINSUM_ALPHABET = "bcdefghijklmnopqrstuvwxyz"


def squish_idcs_up(idcs: str) -> str:
    """
        takes a set of indices like "acv" and maps them to the topmost einsum indices,
        e.g. in this case "xyz"
    """
    sorted_idcs = sorted(idcs)
    return "".join(
        [_EINSUM_ALPHABET[-i - 1] for i in [len(idcs) - 1 - sorted_idcs.index(c) for c in idcs]]
    )


@functools.lru_cache(maxsize=10 ** 6)
def einsum_indices_op(
    m: int, n: int, target_lanes: List[int], batched: bool = False
) -> Tuple[str, str, str]:
    """
        returns einsum indices for the intended operation
        m - operator size
        n - state size
        target_lanes - subset of indices on the state
    """
    assert len(target_lanes) == m, "number of target and operator indices don't match"
    assert all(0 <= lane < n for lane in target_lanes), "target lanes not present in state"

    idcs_op = squish_idcs_up("".join(_EINSUM_ALPHABET[-l - 1] for l in target_lanes)) + "".join(
        _EINSUM_ALPHABET[r] for r in target_lanes
    )
    idcs_target = _EINSUM_ALPHABET[:n]

    assert len(idcs_op) + len(idcs_target) < len(
        _EINSUM_ALPHABET
    ), "too few indices for torch's einsum"

    idcs_result = ""
    idcs_op_lut = dict(
        zip(idcs_op[m:], idcs_op[:m])
    )  # lookup table from operator's right to operator's left indices
    for c in idcs_target:
        if c in idcs_op_lut:
            idcs_result += idcs_op_lut[c]
        else:
            idcs_result += c

    idx_batch = _EINSUM_BATCH_CHARACTER if batched else ""

    return (idcs_op, idx_batch + idcs_target, idx_batch + idcs_result)


def einsum_indices_entrywise(
    m: int, n: int, target_lanes: List[int], batched: bool = False
) -> Tuple[str, str, str]:
    """
        entrywise multiplication variant of einsum_indices
    """
    assert len(target_lanes) == m, "number of target and operator indices don't match"
    assert all(0 <= lane < n for lane in target_lanes), "target lanes not present in state"
    assert n < len(_EINSUM_ALPHABET), "too few indices for torch's einsum"

    idcs_op = "".join(_EINSUM_ALPHABET[r] for r in target_lanes)
    idcs_target = _EINSUM_ALPHABET[:n]
    idcs_result = idcs_target

    idx_batch = _EINSUM_BATCH_CHARACTER if batched else ""
    return (idcs_op, idx_batch + idcs_target, idx_batch + idcs_result)


# FUNDAMENTAL QUANTUM OPERATIONS


def normalize(kob: KetOrBatch) -> KetOrBatch:
    """
        If a batch is given, takes norm of individual vectors
    """
    if is_batch(kob):
        norm = kob.norm(p=2, dim=list(range(1, kob.dim())))
        return mark_batch((kob.transpose(0, -1) / norm).transpose(0, -1))
    else:
        return kob / kob.norm(p=2)


def num_state_qubits(kob: KetOrBatch) -> int:
    return len(kob.shape[1:]) if is_batch(kob) else len(kob.shape)


def num_operator_qubits(op: Operator) -> int:
    assert len(op.shape) % 2 == 0, "operator does not have same input and output indices"
    return len(op.shape) // 2


def probabilities(
    kob: KetOrBatch, measured_lanes: Optional[List[int]] = None, verbose: bool = False
) -> KetOrBatch:
    n = num_state_qubits(kob)
    if measured_lanes is None:
        measured_lanes = range(n)

    idx_batch = _EINSUM_BATCH_CHARACTER if is_batch(kob) else ""
    idcs_kept = "".join(_EINSUM_ALPHABET[i] for i in measured_lanes)
    idcs_einsum = f"{idx_batch}{ _EINSUM_ALPHABET[:n] },{idx_batch}{ _EINSUM_ALPHABET[:n] }->{idx_batch}{idcs_kept}"

    verbose and print(idcs_einsum)

    pvec = torch.einsum(idcs_einsum, kob, kob)
    return mark_batch(pvec.reshape(batch_size(kob), -1)) if is_batch(kob) else pvec.reshape(-1)


def apply(
    op: Operator, kob: KetOrBatch, target_lanes: List[int], verbose: bool = False
) -> KetOrBatch:
    n = num_state_qubits(kob)
    m = num_operator_qubits(op)

    idcs_op, idcs_target, idcs_result = einsum_indices_op(
        m, n, tuple(target_lanes), batched=is_batch(kob)
    )
    idcs_einsum = f"{idcs_op},{idcs_target}->{idcs_result}"
    verbose and print(idcs_einsum)

    return mark_batch_like(kob, torch.einsum(idcs_einsum, op, kob))


def apply_entrywise(
    state_op: Ket, kob: KetOrBatch, target_lanes: List[int], verbose: bool = False
) -> KetOrBatch:
    n = num_state_qubits(kob)
    m = num_state_qubits(state_op)

    idcs_op, idcs_target, idcs_result = einsum_indices_entrywise(
        m, n, tuple(target_lanes), batched=is_batch(kob)
    )
    idcs_einsum = f"{idcs_op},{idcs_target}->{idcs_result}"
    verbose and print(idcs_einsum)

    return mark_batch_like(kob, torch.einsum(idcs_einsum, state_op, kob))


def dot(a: KetOrBatch, b: KetOrBatch) -> KetOrBatch:
    assert is_batch(a) == is_batch(b), "either both or none of the dot arguments should be a batch"

    idx_batch = _EINSUM_BATCH_CHARACTER if is_batch(a) else ""
    idcs_einsum = f"{idx_batch}{ _EINSUM_ALPHABET[:len(a.shape)] },{idx_batch}{ _EINSUM_ALPHABET[:len(b.shape)] }->{idx_batch}"

    return mark_batch_like(a, torch.einsum(idcs_einsum, a, b))


def ctrlMat(op: torch.Tensor, num_control_lanes: int) -> torch.Tensor:
    if num_control_lanes == 0:
        return op
    n = num_operator_qubits(op)
    A = torch.eye(2 ** n)
    AB = torch.zeros(2 ** n, 2 ** n)
    BA = torch.zeros(2 ** n, 2 ** n)
    return ctrlMat(
        torch.cat(
            [torch.cat([A, AB], dim=0), torch.cat([BA, op.reshape(2 ** n, -1)], dim=0)], dim=1
        ).reshape(*[2] * (2 * (n + 1))),
        num_control_lanes - 1,
    )
