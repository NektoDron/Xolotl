"""Opt-in request-local monotonic tracing. No prompt/media bytes are logged."""
from __future__ import annotations
import contextlib
import json
import os
from pathlib import Path
import threading
import time
import uuid

_WRITE_LOCK = threading.Lock()

class TTFTTrace:
    def __init__(self, identifier=None):
        self.path = os.environ.get('M31_TTFT_TRACE')
        self.started_ns = time.perf_counter_ns()
        self.identifier = str(identifier or uuid.uuid4().hex)[:128]
        self.points, self.stages, self.details = {'http_post_entered': 0.0}, {}, {}
        self.finished = False

    def point(self, name):
        self.points.setdefault(name, (time.perf_counter_ns()-self.started_ns)/1e6)

    def add(self, name, milliseconds):
        self.stages[name] = self.stages.get(name, 0.0) + milliseconds

    @contextlib.contextmanager
    def span(self, name):
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            self.add(name, (time.perf_counter_ns()-start)/1e6)

    def finish(self, error=None):
        if self.finished or not self.path:
            return
        self.finished = True
        self.point('request_finished')
        if 'first_generated_delta' in self.points and 'first_content_delta' in self.points:
            self.stages['generated_delta_to_content_ms'] = self.points['first_content_delta']-self.points['first_generated_delta']
        value = dict(format='m31-ttft-trace-v1', id=self.identifier,
                     clock='perf_counter_ns; points are milliseconds from Handler.do_POST entry, after HTTP headers',
                     points_ms=self.points, stages_ms=self.stages, details=self.details)
        if error is not None:
            value['error'] = str(error)
        with _WRITE_LOCK:
            with Path(self.path).open('a') as stream:
                stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')
