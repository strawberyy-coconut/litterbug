"""Tests for the batch-size resolvers.

Both rules exist because of measured failures on this hardware: training at the VRAM-derived
batch peaked at 3.9 GB under AMP, but inference defaults to fp32 and OOMs at that same batch.
"""

from common.config import resolve_batch, resolve_inference_batch


def test_explicit_batch_wins_for_training():
    assert resolve_batch(2) == 2


def test_explicit_batch_wins_for_inference():
    assert resolve_inference_batch(3) == 3


def test_inference_batch_is_at_most_half_the_training_batch():
    # Inference is fp32 where training used AMP, so it needs roughly twice the memory per
    # image. This is the guard that stops the analysis OOMing on the same device training fit.
    assert resolve_inference_batch() <= max(1, resolve_batch() // 2)


def test_inference_batch_is_never_zero():
    assert resolve_inference_batch() >= 1
