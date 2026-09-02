"""Numerical deviation of quantized outputs vs a saved BF16 reference."""
import numpy as np


def openvla_deviation(tokens_q, actions_q, tokens_ref, actions_ref) -> dict:
    """tokens: (R, 7) int64, actions: (R, 7) float (unnormalized units); paired by run index."""
    tokens_q = np.asarray(tokens_q)
    tokens_ref = np.asarray(tokens_ref)
    actions_q = np.asarray(actions_q, dtype=np.float64)
    actions_ref = np.asarray(actions_ref, dtype=np.float64)
    n = min(len(actions_q), len(actions_ref))
    tokens_q, tokens_ref = tokens_q[:n], tokens_ref[:n]
    actions_q, actions_ref = actions_q[:n], actions_ref[:n]
    l2 = np.linalg.norm(actions_q - actions_ref, axis=-1)  # per step
    ref_norm = np.linalg.norm(actions_ref, axis=-1)
    return {
        "token_mismatch_rate": float((tokens_q != tokens_ref).mean()),
        "action_l2_mean": float(l2.mean()),
        "action_l2_std": float(l2.std()),
        "action_l2_relative": float((l2 / np.maximum(ref_norm, 1e-9)).mean()),
        "steps_fully_matching": float((tokens_q == tokens_ref).all(axis=1).mean()),
    }


def smolvla_deviation(actions_q, actions_ref) -> dict:
    """actions: (R, chunk, action_dim) float; paired by run index, trimmed to min length."""
    q = np.asarray(actions_q, dtype=np.float64)
    r = np.asarray(actions_ref, dtype=np.float64)
    n = min(len(q), len(r))  # runs are paired by run_idx (same seed -> same noise/input)
    q, r = q[:n], r[:n]
    l2 = np.linalg.norm(q - r, axis=-1)  # per action in chunk
    ref_norm = np.linalg.norm(r, axis=-1)
    return {
        "chunk_mse": float(((q - r) ** 2).mean()),
        "action_l2_mean": float(l2.mean()),
        "action_l2_std": float(l2.std()),
        "action_l2_relative": float((l2 / np.maximum(ref_norm, 1e-9)).mean()),
        "max_abs_diff": float(np.abs(q - r).max()),
    }
