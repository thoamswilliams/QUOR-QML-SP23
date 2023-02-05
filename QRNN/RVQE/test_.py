import pytest

from .quantum import *


# QUANTUM


def approx_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (a.reshape(-1) - b.reshape(-1)).norm() < 1e-4


def equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.all(a.reshape(-1) == b.reshape(-1)).item()


def test_ket():
    assert equal(ket("00"), tensor([1.0, 0.0, 0.0, 0.0]))
    assert equal(ket("01"), tensor([0.0, 1.0, 0.0, 0.0]))
    assert equal(ket("10"), tensor([0.0, 0.0, 1.0, 0.0]))
    assert equal(ket("11"), tensor([0.0, 0.0, 0.0, 1.0]))
    assert approx_equal(
        ket("0+1"), tensor([0.0000, 0.7071, 0.0000, 0.7071, 0.0000, 0.0000, 0.0000, 0.0000]),
    )
    assert equal(normalize(torch.ones(4).reshape(2, 2)), tensor([0.5, 0.5, 0.5, 0.5]))


def test_probabilities():
    assert equal(probabilities(ket("00"), [0, 1]), tensor([1.0, 0.0, 0.0, 0.0]))
    assert equal(probabilities(ket("11"), [0]), tensor([0.0, 1.0]))
    assert equal(probabilities(ket0(5), [1]), tensor([1.0, 0.0]))

    # EPR pair
    psiminus = normalize(ket("00") - ket("11"))
    assert approx_equal(probabilities(psiminus, [0, 1]), tensor([0.5, 0.0, 0.0, 0.5]))
    assert approx_equal(probabilities(psiminus, [1]), tensor([0.5, 0.5]))


def test_apply():
    assert squish_idcs_up("zkxm") == "zwyx"


def test_batching():
    assert is_batch(mark_batch(ket0(5)))
    assert is_batch(normalize(mark_batch(ket0(2))))
    assert is_batch(ket_to_batch(ket("+0"), copies=3))


# GATE LAYERS
from .gate_layers import *


def test_gates():
    assert equal(XLayer(0)(ket("0")), ket("1"))
    with pytest.raises(AssertionError):
        # wrong index
        assert equal(XLayer(1)(ket("0")), ket("1"))

    assert equal(HLayer(0)(ket("1")), normalize(ket("0") - ket("1")))
    assert approx_equal(rYLayer(0, -pi / 2).to_mat(), tensor([[0.7071, -0.7071], [0.7071, 0.7071]]))
    assert approx_equal(
        crYLayer(0, 1, initial_φ=pi / 2).to_mat(),
        tensor(
            [
                [1.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 1.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.7071, 0.7071],
                [0.0000, 0.0000, -0.7071, 0.7071],
            ]
        ),
    )
    assert approx_equal(
        apply(rYLayer(0, initial_θ=0.33).U, ket0(3), [0], verbose=True),
        apply(rYLayer(2, initial_θ=0.33).U, ket0(3), [2], verbose=True).transpose(0, 2),
    )
    assert equal(
        crYLayer(0, 1).forward(ket("00")), ket("00")
    ), "controlled gate has no action if control is 0"
    assert not equal(
        crYLayer(1, 0).forward(ket("0+")), ket("0+")
    ), "controlled gate acts if control is 1"
    foo = apply(
        crYLayer([0], 1, initial_φ=1.0).U,
        apply(HLayer(0).U, ket0(2), [0], verbose=True),
        [1, 0],
        verbose=True,
    )
    assert approx_equal(
        foo, tensor([0.7071, 0.0000, 0.7071, 0.0000])
    ), "crYLayer has no action if control is 0"
    assert equal(
        cmiYLayer(0, 1).to_mat(),
        tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0, 0.0],
            ]
        ),
    ), "controlled-iY"
    foo = tensor([[1, 2], [3, 4]], dtype=torch.float)
    assert equal(cmiYLayer(0, 1)(foo), tensor([1.0, 2.0, -4.0, 3.0]))
    assert equal(cmiYLayer(0, 1).T.forward(foo), tensor([1.0, 2.0, 4.0, -3.0]))


def test_gates_batched():
    batch = ket_to_batch(ket("++0"), copies=10)
    assert is_batch(XLayer(0)(batch))
    assert is_batch(HLayer(0)(batch))
    assert is_batch(rYLayer(0, -pi / 2)(batch))
    assert is_batch(cmiYLayer(0, 1).T.forward(batch))


def test_parameter_sharing():
    param1 = nn.Parameter(torch.tensor(0.1))
    param2 = nn.Parameter(torch.tensor(0.2))
    foo = rYLayer(2, initial_θ=param1)
    assert (
        len(list(nn.Sequential(foo, foo.T).named_parameters())) == 1
    ), "we expect there to be only one parameter"
    assert (
        len(list(nn.Sequential(foo, rYLayer(2, initial_θ=param2)).named_parameters())) == 2
    ), "there have to be two separate parameters here"


def test_backwards_call():
    temp = PostselectLayer(0, 0).forward(crYLayer(1, 2).T.forward(ket0(3)))[0][1][0]
    temp.backward()


def test_compatibility_with_qiskit():
    psi = normalize(tensor([1.0, 0.0, 1, 1.0, 0, 0.0, 0, 1.0]).reshape(2, 2, 2))
    assert equal(
        probabilities(psi),
        tensor([0.2500, 0.0000, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000, 0.2500]),
    )

    layer1 = crYLayer([1], 2, initial_φ=pi / 4)
    layer2 = rYLayer(2, initial_θ=pi / 8)
    layer3 = cmiYLayer(2, 0)

    psi = layer1(psi)
    assert approx_equal(
        psi, tensor([0.5000, 0.0000, 0.2706, 0.6533, 0.0000, 0.0000, -0.1913, 0.4619])
    )
    psi = layer2(psi)
    assert approx_equal(
        psi, tensor([0.4904, 0.0975, 0.1379, 0.6935, 0.0000, 0.0000, -0.2778, 0.4157])
    )
    psi = layer3(psi)
    assert approx_equal(
        psi, tensor([0.4904, 0.0000, 0.1379, -0.4157, 0.0000, 0.0975, -0.2778, 0.6935])
    )
    psi = layer2.T(psi)
    assert approx_equal(
        psi, tensor([0.4810, -0.0957, 0.0542, -0.4347, 0.0190, 0.0957, -0.1371, 0.7344])
    )
    psi = layer1.T(psi)
    assert approx_equal(
        psi, tensor([0.4810, -0.0957, -0.1163, -0.4223, 0.0190, 0.0957, 0.1543, 0.7310])
    )
    psi = PostselectLayer(2, on=0).forward(psi)
    assert approx_equal(
        probabilities(psi),
        tensor([0.8599, 0.0000, 0.0502, 0.0000, 0.0013, 0.0000, 0.0885, 0.0000]),
    )


# Quantum Neuron Layer
def test_slow_fast_qn_equivalence(capsys):
    # test that slow and fast variant produce same results
    workspace = [0, 1, 2, 3, 4]
    outlane = 2
    order = 2
    r0 = QuantumNeuronLayer(
        workspace=workspace,
        outlane=outlane,
        ancillas=list(range(len(workspace), len(workspace) + order)),
        degree=2,
    )
    r1 = FastQuantumNeuronLayer(workspace=workspace, outlane=outlane, order=order, degree=2)
    θ, φ = torch.rand(2)
    for name, p in r0.named_parameters():
        if name[-1:] == "θ":  # rY
            nn.init.constant_(p, θ)
        elif name[-1:] == "φ":  # crY
            nn.init.constant_(p, φ)
    for name, p in r1.named_parameters():
        if name[-1:] == "θ":  # rY
            nn.init.constant_(p, θ)
        elif name[-1:] == "φ":  # crY
            nn.init.constant_(p, φ)

    # prepare random state with same marginal
    unitary = UnitaryLayer(workspace)
    for p in unitary.parameters():
        torch.nn.init.uniform_(p)

    psi0 = unitary(ket0(len(workspace) + order))
    psi1 = unitary(ket0(len(workspace)))
    assert equal(psi0.reshape(-1)[:: 2 ** order], psi1.reshape(-1))

    # apply slow and fast neurons and compare marginal states (as we know the ancillas for the slow one will be 0)
    psi0 = r0(psi0)
    psi1 = r1(psi1)
    assert approx_equal(psi0.reshape(-1)[:: 2 ** order], psi1.reshape(-1))


# RVQE
from .model import *
import sys


def test_rvqe_cell():
    assert (
        RVQECell(workspace_size=4, input_size=1, stages=1, order=2, fast=False).num_qubits
        == 4 + 1 + 2
    )
    temp = RVQECell(workspace_size=4, input_size=1, stages=1, order=2, fast=False).forward(
        ket_to_batch(ket0(7), copies=1), torch.LongTensor([[1]])
    )[0][0][0]
    temp.backward()


def test_fast_rvqe_cell():
    assert (
        RVQECell(workspace_size=4, input_size=1, stages=1, order=2, fast=True).num_qubits
        == 4 + 1 + 0
    )
    temp = RVQECell(workspace_size=4, input_size=1, stages=1, order=2, fast=True).forward(
        ket_to_batch(ket0(5), copies=1), torch.LongTensor([[1]])
    )[0][0][0]
    temp.backward()


def test_rvqe():
    for fast in [True, False]:
        rvqe = RVQE(workspace_size=2, input_size=1, stages=1, order=2, fast=fast)
        probs, measured_seq, min_ps_prob = rvqe(
            tensor(
                [[[0], [1], [1], [1], [1]], [[1], [1], [1], [1], [1]], [[0], [0], [0], [0], [0]]]
            ),
            tensor(
                [[[1], [1], [0], [0], [0]], [[1], [1], [0], [0], [0]], [[0], [0], [0], [0], [0]]]
            ),
            postselect_measurement=False,
        )
        assert min_ps_prob == 1.0
        assert probs.shape == torch.Size([3, 2, 4])  # BATCH x CLASS x LENGTH
        assert measured_seq.shape == torch.Size([3, 4, 1])  # BATCH x LENGTH x Bitvector Size


def test_rvqe_batching(capsys):
    # batch size = 3
    # sequence length = 5
    # width = 1
    sentences = tensor(
        [[[1], [0], [1], [1], [1]], [[0], [0], [0], [1], [1]], [[1], [1], [0], [0], [1]]]
    )
    targets = tensor(
        [[[1], [0], [0], [0], [0]], [[1], [1], [0], [0], [0]], [[1], [1], [1], [0], [0]]]
    )
    for fast in [True, False]:
        # in-built batch
        rvqe = RVQE(workspace_size=2, input_size=1, stages=1, order=2, fast=fast)
        probs_batch, measured_seq_batch, _ = rvqe(sentences, targets, postselect_measurement=True)

        # manual batch
        probs = []
        measured_seq = []
        for sentence, target in zip(sentences, targets):
            p, m, _ = rvqe(sentence, target, postselect_measurement=True)
            probs.append(p)
            measured_seq.append(m)
        probs_manual = torch.stack(probs)
        measured_seq_manual = torch.stack(measured_seq)

        # compare
        assert approx_equal(probs_batch, probs_manual)
        assert equal(measured_seq_batch, measured_seq_manual)


# DATA
from .data import *


def test_data():
    assert equal(
        bitword_to_onehot(tensor([0, 0, 1]), 3), tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    assert bitword_to_int(tensor([0, 1, 1])) == 3
    assert int_to_bitword(12, 10) == [0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
    assert bitword_to_str([1, 1, 1, 0]) == "1110"
    assert bitword_to_str(char_to_bitword("c", "abc", 3)) == "010"
    assert bitword_to_str(char_to_bitword("c", "abc", 1)) == "0"
