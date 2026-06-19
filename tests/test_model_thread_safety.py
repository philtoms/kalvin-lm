"""Thread-safety tests for Model / STM (KB-305).

Two test groups, plus a documented known-limitation (xfail):

1. ``TestConcurrentAccess`` (no NLP marker) — a focused stress test on the
   data structures alone. A writer thread mutates the model while a reader
   thread exercises every read path (``klines``, ``find``, ``unpack``,
   ``iter_stm``, iteration). It must complete without raising and without
   hanging. **This is the reliable guard for what the KB-305 locks actually
   fix: the crash/atomicity class** (iterator-mutation ``RuntimeError``,
   deadlock, stale mid-write reads). It uses only ``Model``/``STM``/``KLine``
   — no tokenizer — so it is NOT gated behind ``requires_nlp_data`` and runs
   on every clone / in CI.

2. ``TestS1CountStableUnderBoundedReads`` (``@requires_nlp_data``) — a narrow
   *positive* characterisation: in a fully-drained scenario, *fast bounded*
   reads (``unpack``/``find``, no full ``klines()`` scan) do not perturb the
   Cogitator's timing enough to change the S1 frame count. This guards the
   invariant the locks *can* provide (per-operation atomicity) for that
   restricted read shape.

3. ``TestS1CountUnderVerboseReads`` (``@requires_nlp_data``, ``xfail``) — the
   **honest regression for the reported 24-vs-22 bug**. It reproduces the real
   ``scripts/kalvin_test.py --verbose`` subscriber (``kline_display`` →
   ``model.klines()`` + ``unpack``) and asserts the S1 count equals the
   no-read baseline. This **fails** and is marked ``xfail(strict=False)``:
   the locks make every model operation atomic, but they CANNOT make the
   rationalisation event count identical across subscriber read patterns,
   because the background Cogitator processes work items asynchronously
   relative to the main-thread ``rationalise`` loop and ``expand()``'s result
   depends on the model state at processing time. (A bare ``time.sleep``
   subscriber — no model access — also moves the count, proving the cause is
   timing, not model-consistency.) Count-determinism is owned by KB-347; this
   test will flip to *xpass* when KB-347 lands and should then be promoted to
   a strict assertion.
"""

from __future__ import annotations

import threading
import time

import pytest

from kalvin.kline import KLine
from kalvin.model import Model

# NOTE: ``requires_nlp_data`` is applied per-class below, not module-wide, so
# that the pure-data-structure stress test (TestConcurrentAccess) always runs.
from tests.conftest import requires_nlp_data

# MHALL-SVO curriculum (duplicated inline — scripts/ is not on the test path).
SOURCE = """
(Mary had a little lamb)
MHALL == SVO =>
   S(ubject) = M
   V(erb) = H
   O(bject) = ALL =>
     A > D(et)
     L > M(od)
     L > O
"""


# ── 1. Concurrent reader/writer stress (no NLP data required) ─────────


class TestConcurrentAccess:
    """KB-305: concurrent readers + writers must not raise or deadlock.

    Exceptions raised inside a worker thread do NOT propagate to the joiner, so
    each worker captures them in a shared list and the main thread asserts it
    is empty. ``join(timeout=...)`` returns silently on timeout, so the main
    thread asserts ``not t.is_alive()`` afterwards — a genuine deadlock would
    otherwise make this test pass vacuously.
    """

    def test_concurrent_readers_and_writers_no_exception(self):
        m = Model(stm_bound=4096)
        errors: list[tuple[str, BaseException]] = []
        stop = threading.Event()

        def writer() -> None:
            try:
                i = 0
                while not stop.is_set():
                    sig = ((i * 7) + 3) % 64 or 1
                    m.add_to_frame(KLine(sig, [(i % 8) + 1]))
                    if i % 3 == 0:
                        ltm_sig = (sig * 5) % 64 or 2
                        m.add_to_ltm(KLine(ltm_sig, [(i % 4) + 1]))
                    i += 1
            except BaseException as exc:  # noqa: BLE001 — surface in main thread
                errors.append(("writer", exc))

        def reader() -> None:
            try:
                while not stop.is_set():
                    # Exercise every read path. unpack() raises ValueError for
                    # non-decomposable klines (the common case here) — swallow it.
                    for kl in m.klines():
                        m.find(kl.signature)
                        try:
                            m.unpack(kl)
                        except ValueError:
                            pass
                    list(m)  # Model.__iter__ snapshot
                    len(m)
                    m.where(7)
                    list(m.iter_stm())  # STM snapshot, materialise fully
            except BaseException as exc:  # noqa: BLE001 — surface in main thread
                errors.append(("reader", exc))

        t_writer = threading.Thread(target=writer, name="kb305-writer")
        t_reader = threading.Thread(target=reader, name="kb305-reader")
        t_writer.start()
        t_reader.start()

        time.sleep(0.8)
        stop.set()

        t_writer.join(timeout=5.0)
        t_reader.join(timeout=5.0)

        # Fail loudly on hang/deadlock — join(timeout) returns silently.
        assert not t_writer.is_alive(), "writer thread hung (possible deadlock)"
        assert not t_reader.is_alive(), "reader thread hung (possible deadlock)"
        assert not errors, f"worker threads raised: {errors!r}"


# ── 2. S1-count stability under model reads (requires NLP data) ───────


def _load_kalvin_test_module():
    """Import ``scripts/kalvin_test.py`` (not on the test path) for its helpers.

    Used only by the verbose-path repro (gated by ``requires_nlp_data``) so
    that it reproduces the exact ``kline_display`` read shape rather than a
    hand-rolled approximation.
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "kalvin_test.py"
    spec = importlib.util.spec_from_file_location("kalvin_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _significance_level(sig: int) -> str:
    """Classify a raw significance value into S1–S4 (mirrors kalvin_test)."""
    from kalvin.expand import D_MAX, boundaries, classify

    if sig == D_MAX - 1:
        return "S1"
    if sig == 0:
        return "S4"
    s12, s23, s34 = boundaries()
    return classify(sig, s12, s23, s34)


def _run_agent(source: str, *, read_mode: str) -> int:
    """Rationalise *source* and return the S1 frame-event count.

    *read_mode* selects what the ``on_event`` subscriber does on each event:

    - ``"none"``    — no model access (the baseline).
    - ``"bounded"`` — fast bounded reads (``unpack``/``find`` only). Too cheap
      to perturb the Cogitator's timing, so the count matches ``"none"`` in a
      drained scenario (``TestS1CountStableUnderBoundedReads``).
    - ``"verbose"`` — the real ``scripts/kalvin_test.py --verbose`` path
      (``kline_display`` → ``model.klines()`` + ``unpack``). Holds the Model
      lock long enough to change the Cogitator's interleaving, so the count
      differs from ``"none"`` — the reported 24-vs-22/23 bug. Root cause
      (async-Cogitator count-determinism) is owned by KB-347.

    Uses ``cogitate_drain`` (returns as soon as the backlog empties — far
    faster and more deterministic than the 2s idle "done" event) and always
    ``cogitate_join`` in a ``finally`` so no daemon thread lingers to fire
    callbacks into a later run.
    """
    from kalvin.agent import KAgent
    from kalvin.events import EventBus
    from kalvin.nlp_tokenizer import NLPTokenizer
    from ks.compiler import compile_source

    tokenizer = NLPTokenizer.from_files()
    klines = compile_source(source, tokenizer, dev=True)

    bus = EventBus()
    agent = KAgent(tokenizer=tokenizer, adapter=bus)

    verbose_display = None
    if read_mode == "verbose":
        verbose_display = _load_kalvin_test_module().kline_display

    counter = {"s1": 0}
    counter_lock = threading.Lock()  # on_event runs on both threads.

    def on_event(event) -> None:
        if event.kind == "done":
            return
        # S1 count excludes "ground" events (matches scripts/kalvin_test.py).
        if _significance_level(event.significance) == "S1" and event.kind != "ground":
            with counter_lock:
                counter["s1"] += 1
        if read_mode == "bounded":
            # Bounded read mirroring kline_display's decode access pattern,
            # WITHOUT the O(n) model.klines() scan. unpack() walks the graph
            # and raises ValueError fast for the common non-decomposable
            # payloads. This shape is too cheap to perturb timing, so it is a
            # characterization of per-operation atomicity, not a faithful
            # repro of the verbose bug (see the "verbose" mode for that).
            try:
                agent.model.unpack(event.query)
            except ValueError:
                pass
            try:
                agent.model.unpack(event.proposal)
            except ValueError:
                pass
            _ = agent.model.find(event.query.signature)
        elif read_mode == "verbose":
            # The real --verbose decode path: kline_display scans
            # model.klines() and unpacks every candidate. This read shape is
            # what perturbs the Cogitator's timing (the reported bug).
            verbose_display(event.query, tokenizer, agent.model)
            verbose_display(event.proposal, tokenizer, agent.model)

    bus.subscribe(on_event)

    try:
        for k in klines:
            agent.rationalise(k)
        drained = agent.cogitate_drain(timeout=60.0)
        assert drained, "cogitator did not drain within timeout"
    finally:
        # Stop the background thread so it cannot fire "done"/events into a
        # subsequent _run_agent invocation.
        agent.cogitate_join(timeout=30.0)

    return counter["s1"]


@requires_nlp_data
class TestS1CountStableUnderBoundedReads:
    """Narrow characterization: *bounded* reads don't change the drained count.

    Guards the invariant the KB-305 locks *can* provide (per-operation
    atomicity) for fast, bounded reads in a fully-drained scenario. This is
    NOT a faithful reproduction of the reported 24-vs-22 bug — that lives in
    ``TestS1CountUnderVerboseReads`` (xfail). See the module docstring for why
    locking alone cannot make the count identical across read shapes.
    """

    def test_s1_count_stable_under_bounded_reads_in_drained_scenario(self):
        n_noop = _run_agent(SOURCE, read_mode="none")
        n_bounded = _run_agent(SOURCE, read_mode="bounded")
        assert n_noop == n_bounded, (
            f"S1 frame count changed under bounded reads in a drained "
            f"scenario: none={n_noop} vs bounded={n_bounded}. (Bounded reads "
            f"should be too cheap to perturb the Cogitator's timing; if this "
            f"fails, a public read method may have become non-atomic.)"
        )
        # Sanity: the curriculum is expected to produce S1 frame events.
        assert n_noop > 0


@requires_nlp_data
class TestS1CountUnderVerboseReads:
    """Faithful repro of the reported 24-vs-22 bug (PENDING KB-347).

    The real ``--verbose`` subscriber (``kline_display`` → ``model.klines()`` +
    ``unpack``) changes the S1 frame count versus a no-read baseline. The
    KB-305 locks make every model operation atomic but cannot make the
    rationalisation count identical across subscriber read patterns: the
    background Cogitator processes work items asynchronously relative to the
    main-thread ``rationalise`` loop, and ``expand()``'s result depends on the
    model state at processing time. Count-determinism is owned by KB-347.
    """

    @pytest.mark.xfail(
        strict=False,
        reason=(
            "KB-347: async-Cogitator event-count determinism. Locking makes "
            "each model op atomic but cannot make the S1 count identical "
            "across subscriber read patterns. Two independent effects, both "
            "outside Model/STM scope: (1) the verbose kline_display reads are "
            "expensive enough that scripts/kalvin_test.py's 5s 'done'-event "
            "wait expires mid-cogitation, reporting a PARTIAL count (22; "
            "rising to 23 with a longer wait); (2) even at full completion the "
            "count is timing-dependent — a bare time.sleep subscriber (no "
            "model access) also moves it. Promote to a strict assertion once "
            "KB-347 lands."
        ),
    )
    def test_s1_count_not_stable_under_verbose_reads__pending_kb347(self):
        n_noop = _run_agent(SOURCE, read_mode="none")
        n_verbose = _run_agent(SOURCE, read_mode="verbose")
        # Observed (post-KB-305 locking): none=24; verbose=23 at full drain,
        # 22 in the full scripts/kalvin_test.py --verbose flow (5s done-timeout
        # partial). Either way it differs from the no-read baseline.
        assert n_noop == n_verbose, (
            f"S1 frame count differs under verbose reads: none={n_noop} vs "
            f"verbose={n_verbose}. (Expected to fail until KB-347; if this "
            f"now passes, KB-347 may be fixed — promote this assertion.)"
        )
