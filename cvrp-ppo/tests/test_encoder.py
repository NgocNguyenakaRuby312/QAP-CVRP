"""
tests/test_encoder.py — Encoder shape, norm, feature, gradient, and device checks.
Run: python tests/test_encoder.py   OR   pytest tests/test_encoder.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from encoder import FullEncoder

B, N = 4, 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_state():
    coords = torch.rand(B, N + 1, 2, device=device)
    demands = torch.zeros(B, N + 1, device=device)
    demands[:, 1:] = torch.randint(1, 10, (B, N), device=device).float()
    return {
        "coords":   coords,
        "demands":  demands,
        "capacity": torch.full((B,), 30.0, device=device),
    }


def test_shapes():
    enc = FullEncoder(5, 2, 16, 5).to(device)
    state = _make_state()
    psi_prime, features, knn_indices = enc(state)
    assert features.shape == (B, N + 1, 5), f"features: {features.shape}"
    assert psi_prime.shape == (B, N + 1, 2), f"psi_prime: {psi_prime.shape}"
    assert knn_indices.shape == (B, N + 1, 5), f"knn: {knn_indices.shape}"


def test_unit_norm_after_projection():
    enc = FullEncoder(5, 2, 16, 5).to(device)
    state = _make_state()
    features = enc.feature_builder(state)
    psi = enc.qap_encoder.amplitude_proj(features)
    norms = psi.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        f"psi norms: min={norms.min():.6f} max={norms.max():.6f}"


def test_unit_norm_after_rotation():
    enc = FullEncoder(5, 2, 16, 5).to(device)
    state = _make_state()
    psi_prime, _, _ = enc(state)
    norms = psi_prime.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        f"psi_prime norms: min={norms.min():.6f} max={norms.max():.6f}"


def test_feature_order():
    enc = FullEncoder(5, 2, 16, 5).to(device)
    state = _make_state()
    features = enc.feature_builder(state)
    # Depot demand ratio = 0
    assert features[:, 0, 0].abs().max() == 0, "depot demand ratio must be 0"
    # Angle in [-1, 1]
    assert features[:, :, 4].abs().max() <= 1.0, "angle must be in [-1, 1]"


def test_gradient_flow():
    enc = FullEncoder(5, 2, 16, 5).to(device)
    state = _make_state()
    psi_prime, _, _ = enc(state)
    loss = psi_prime.sum()
    loss.backward()
    for name, p in enc.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert not torch.isnan(p.grad).any(), f"NaN grad for {name}"


def test_device():
    enc = FullEncoder(5, 2, 16, 5).to(device)
    state = _make_state()
    psi_prime, _, _ = enc(state)
    assert psi_prime.device.type == device.type, \
        f"psi_prime on {psi_prime.device}, expected {device}"


if __name__ == "__main__":
    print(f"Device: {device}")
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    if device.type == "cuda":
        print(f"VRAM used: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
