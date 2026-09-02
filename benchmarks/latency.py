"""Per-phase CUDA timing via forward hooks.

PhaseProfiler accumulates CUDA-event elapsed times across one model step
(`reset()` before the step, `finish()` after a synchronize at step end).
HookTimer attaches forward hooks to a module; `classify(args, kwargs)`
maps each call to a phase name (or None to ignore the call).
"""
import torch


class PhaseProfiler:
    def __init__(self):
        self.pending = []  # (name, start_event, end_event)
        self.sums = {}

    def reset(self):
        self.pending = []
        self.sums = {}

    def add(self, name, start_ev, end_ev):
        self.pending.append((name, start_ev, end_ev))

    def finish(self) -> dict:
        torch.cuda.synchronize()
        for name, s, e in self.pending:
            self.sums[name] = self.sums.get(name, 0.0) + s.elapsed_time(e)
        self.pending = []
        return dict(self.sums)


class HookTimer:
    """Records CUDA events around each forward of `module`.

    classify(args, kwargs) -> phase name; return None to skip the call.
    """

    def __init__(self, profiler: PhaseProfiler, name: str, classify=None):
        self.profiler = profiler
        self.name = name
        self.classify = classify
        self._start = None
        self._cur = None
        self.handles = []

    def _pre(self, module, args, kwargs):
        phase = self.name if self.classify is None else self.classify(args, kwargs)
        self._cur = phase
        if phase is None:
            return
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._start = ev

    def _post(self, module, args, output):
        if self._cur is None or self._start is None:
            return
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self.profiler.add(self._cur, self._start, ev)
        self._start = None
        self._cur = None

    def attach(self, module):
        self.handles.append(module.register_forward_pre_hook(self._pre, with_kwargs=True))
        self.handles.append(module.register_forward_hook(self._post))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def seq_len_of(args, kwargs) -> int | None:
    """Best-effort sequence length of a decoder forward call."""
    names = ["input_ids", "inputs_embeds", "input_embeds"]
    for n in names:
        v = kwargs.get(n)
        if v is None:
            continue
        if isinstance(v, torch.Tensor):
            return v.shape[-2] if v.dim() >= 3 else v.shape[-1]
        if isinstance(v, list):  # e.g. inputs_embeds=[prefix, None]
            v = v[0] if v[0] is not None else (v[1] if len(v) > 1 and v[1] is not None else None)
            if isinstance(v, torch.Tensor):
                return v.shape[-2]
    return None
