import torch
from torch import nn
from torch.optim import SGD

from utils.checkpoint import CheckpointIO


def _train_step(model, opt):
    x = torch.randn(2, 2)
    loss = model(x).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()


def test_save_and_load(tmp_path):
    ckpt = CheckpointIO(str(tmp_path))
    torch.manual_seed(0)
    model = nn.Linear(2, 2)
    opt = SGD(model.parameters(), lr=0.1)
    _train_step(model, opt)
    ckpt.save(model=model, optimizers=[opt], step=1)
    manifest = ckpt.resolve_latest_or_none()
    assert manifest is not None

    model2 = nn.Linear(2, 2)
    opt2 = SGD(model2.parameters(), lr=0.1)
    info = ckpt.load(manifest, model2, map_location="cpu", optimizers=[opt2])
    assert info["step"] == 1
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)


def test_deterministic_resume(tmp_path):
    ckpt = CheckpointIO(str(tmp_path))

    # reference run
    torch.manual_seed(0)
    model_ref = nn.Linear(2, 2)
    opt_ref = SGD(model_ref.parameters(), lr=0.1)
    for _ in range(3):
        _train_step(model_ref, opt_ref)
    ref_params = [p.detach().clone() for p in model_ref.parameters()]

    # interrupted run
    torch.manual_seed(0)
    model = nn.Linear(2, 2)
    opt = SGD(model.parameters(), lr=0.1)
    for step in range(1, 3):
        _train_step(model, opt)
        if step == 2:
            ckpt.save(model=model, optimizers=[opt], step=step)

    # load and continue
    model2 = nn.Linear(2, 2)
    opt2 = SGD(model2.parameters(), lr=0.1)
    manifest = ckpt.resolve_latest_or_none()
    info = ckpt.load(manifest, model2, map_location="cpu", optimizers=[opt2])
    for step in range(info["step"] + 1, 3 + 1):
        _train_step(model2, opt2)

    for p_ref, p in zip(ref_params, model2.parameters()):
        assert torch.allclose(p_ref, p)
