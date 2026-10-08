# Copyright 2026 verl contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Opt-in FP8 reload timing without synchronizing individual weights or buckets."""

import functools
import json
import os
import socket
import time
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar

import torch

_current = ContextVar("weight_update_profile", default=None)


def enabled():
    return os.environ.get("VERL_FP8_UPDATE_PROFILE", "0") == "1"


def emit(event, **fields):
    if enabled():
        print(
            "VERL_FP8_PROFILE "
            + json.dumps(
                {"event": event, "hostname": socket.gethostname(), "pid": os.getpid(), **fields},
                sort_keys=True,
            ),
            flush=True,
        )


def profile_export(weights, step, rank):
    """Time lazy Megatron/HF export work, excluding time spent consuming yields."""
    iterator = iter(weights)
    wall_ms = 0.0
    payload_bytes = 0
    tensor_count = 0
    dtypes = defaultdict(int)
    while True:
        begin = time.perf_counter()
        try:
            name, tensor = next(iterator)
        except StopIteration:
            wall_ms += (time.perf_counter() - begin) * 1000
            break
        wall_ms += (time.perf_counter() - begin) * 1000
        payload_bytes += tensor.nbytes
        tensor_count += 1
        dtypes[str(tensor.dtype)] += tensor.nbytes
        yield name, tensor
    emit(
        "actor_export",
        step_id=step,
        rank=rank,
        export_iterator_wall_ms=wall_ms,
        payload_bytes=payload_bytes,
        tensor_count=tensor_count,
        dtype_bytes=dict(dtypes),
    )


class ReloadProfile:
    def __init__(self, step, rank):
        self.step = step
        self.rank = rank
        self.wall_ms = defaultdict(float)
        self.calls = defaultdict(int)
        self.events = []
        self.input_bytes = 0
        self.fp8_bytes = 0
        self.scale_bytes = 0
        self.quantized_tensors = 0

    @contextmanager
    def stage(self, name):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin = time.perf_counter()
        start.record()
        try:
            yield
        finally:
            end.record()
            self.events.append((name, start, end))
            self.wall_ms[name] += (time.perf_counter() - begin) * 1000
            self.calls[name] += 1

    def finish(self, total_ms):
        # A single end-of-publication fence; never fence a conversion or bucket.
        fence = torch.cuda.Event()
        fence.record()
        begin = time.perf_counter()
        fence.synchronize()
        fence_ms = (time.perf_counter() - begin) * 1000
        gpu_ms = defaultdict(float)
        for name, start, end in self.events:
            gpu_ms[name] += start.elapsed_time(end)
        emit(
            "vllm_fp8_reload",
            step_id=self.step,
            rank=self.rank,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            nccl_ib_hca=os.environ.get("NCCL_IB_HCA", ""),
            wall_ms=dict(self.wall_ms),
            gpu_stream_ms=dict(gpu_ms),
            calls=dict(self.calls),
            total_wall_ms=total_ms,
            profile_fence_ms=fence_ms,
            input_quantized_bytes=self.input_bytes,
            output_fp8_bytes=self.fp8_bytes,
            output_scale_bytes=self.scale_bytes,
            quantized_tensors=self.quantized_tensors,
            note="load is inclusive of conversion; stream intervals can include host launch gaps",
        )


@contextmanager
def stage(name):
    profile = _current.get()
    if profile is None:
        yield
    else:
        with profile.stage(name):
            yield


def count_quantized(source, quantized, scales):
    profile = _current.get()
    if profile is not None:
        profile.input_bytes += source.nbytes
        profile.fp8_bytes += quantized.nbytes
        profile.scale_bytes += scales.nbytes
        profile.quantized_tensors += 1


def profile_reload(fn):
    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        if not enabled():
            return fn(self, *args, **kwargs)
        profile = ReloadProfile(kwargs.get("global_steps"), getattr(self, "rank", self.local_rank))
        token = _current.set(profile)
        begin = time.perf_counter()
        try:
            result = fn(self, *args, **kwargs)
            profile.finish((time.perf_counter() - begin) * 1000)
            return result
        finally:
            _current.reset(token)

    return wrapped
