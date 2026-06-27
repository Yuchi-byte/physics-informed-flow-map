"""The 0002 teacher (esd_teacher) distillation path: the off-diagonal target comes from a
frozen teacher while the diagonal stays on data. Unconditional only."""

import copy

import torch
from torch.utils.data import DataLoader, TensorDataset

from physics_informed_flow_map.flow_matching.models import (
    DiTModelConfig,
    MLPModelConfig,
    activate_x_cond_conditioning,
    build_model,
)
from physics_informed_flow_map.flow_matching.train import train

SHAPE = (2,)


def _loader() -> DataLoader:
    x = torch.randn(64, *SHAPE)
    labels = torch.zeros(
        64, dtype=torch.long
    )  # unconditional; train() maps these to None
    return DataLoader(TensorDataset(x, labels), batch_size=16, shuffle=True)


def test_esd_teacher_trains_and_freezes_teacher() -> None:
    device = torch.device("cpu")
    student = build_model(SHAPE, 0, MLPModelConfig(width=16, depth=2)).to(device)
    teacher = build_model(SHAPE, 0, MLPModelConfig(width=16, depth=2)).to(device)
    teacher.load_state_dict(student.state_dict())
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher_snapshot = copy.deepcopy(teacher.state_dict())

    history, _ = train(
        student,
        _loader(),
        n_epochs=2,
        lr=1e-3,
        device=device,
        num_classes=0,
        distillation_type="esd_teacher",
        loss_weighting="adaptive",
        flow_map_warmup_steps=0,  # off-diagonal (teacher-distilled) on immediately
        flow_map_anneal_end=1,
        t_cond_0_rate=0.1,
        t_cond_power=2.0,
        teacher_model=teacher,
    )

    assert history, "no training steps ran"
    assert all(torch.isfinite(torch.tensor(h["total"])) for h in history)
    # The teacher must stay frozen — distillation reads it, never updates it.
    for k, v in teacher.state_dict().items():
        assert torch.equal(v, teacher_snapshot[k]), f"teacher param {k} changed"


def test_activate_x_cond_connects_pathway() -> None:
    """The fix for mfm's dead x_cond init: copy x_embedder -> x_cond_embedder."""
    m = build_model(
        (1, 16, 16), 0, DiTModelConfig(hidden=64, depth=2, num_heads=4, patch_size=4)
    )
    dit = m.model.dit
    assert dit.x_cond_embedder.proj.weight.norm() == 0  # mfm zero-inits it (dead)
    assert activate_x_cond_conditioning(m) is True
    assert torch.equal(dit.x_cond_embedder.proj.weight, dit.x_embedder.proj.weight)
    assert dit.x_cond_embedder.proj.weight.norm() > 0
    # The MLP has no x_cond pathway, so activation is a no-op.
    mlp = build_model((2,), 0, MLPModelConfig(width=16, depth=2))
    assert activate_x_cond_conditioning(mlp) is False


def test_scratch_mf_needs_no_teacher() -> None:
    """The from-scratch mf path still runs with teacher_model=None (regression guard)."""
    device = torch.device("cpu")
    student = build_model(SHAPE, 0, MLPModelConfig(width=16, depth=2)).to(device)
    history, _ = train(
        student,
        _loader(),
        n_epochs=1,
        lr=1e-3,
        device=device,
        num_classes=0,
        distillation_type="mf",
        loss_weighting="adaptive",
        flow_map_warmup_steps=0,
        flow_map_anneal_end=1,
    )
    assert history
