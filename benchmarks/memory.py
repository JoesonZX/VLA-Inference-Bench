import torch


def allocated_gb() -> float:
    return torch.cuda.memory_allocated() / 2**30


def reserved_gb() -> float:
    return torch.cuda.memory_reserved() / 2**30


def reset_peak():
    torch.cuda.reset_peak_memory_stats()


def peak_snapshot() -> dict:
    return {
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 4),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 4),
    }
