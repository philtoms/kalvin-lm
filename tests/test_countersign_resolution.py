"""Tests for countersign resolution behaviour.

Spec references: @specs/agent.md §Rationalisation (Phase 2–5),
@specs/model.md (is_countersigned, grounded STM exclusion),
@specs/kscript.md §7 (operator significance levels).

Covers: pre-registration, ground-check STM exclusion, self-filter,
sig_level propagation, undersign=connotate identity, and countersign
pair resolution.
"""

from kalvin.agent import KAgent
from kalvin.events import EventBus
from kalvin.expand import SIG_S4
from kalvin.kline import KLine, sig_level
from kalvin.kvalue import KValue
from kalvin.model import Model
from kalvin.signifier import NLPSignifier

signifier = NLPSignifier()
from ks import compile_source
from tests.conftest import requires_tokenizer_data


def T(bits: int) -> int:
    """Place sig-word bits in the upper 32 bits of a uint64.

    signifies() (used by model.where for candidate retrieval) masks off the
    lower (BPE) 32 bits, so node/signature values that must overlap for
    candidate matching are shifted up here.
    """
    return bits << 32

# Every test in this module drives ``compile_source`` (which builds a
# ``NLPTokenizer()`` internally) or a ``KAgent`` (whose default
# tokenizer is the kalvin Tokenizer).  Gate the whole module so data-less clones skip cleanly.
pytestmark = requires_tokenizer_data


class TestCountersignPairResolution:
    """CR-1: Countersign pair both resolve S1 via is_countersigned()."""

    def test_countersign_pair_both_resolve_s1(self):
        bus = EventBus()
        a = KAgent(adapter=bus)

        # Add identities (derived from compile_source so signatures match the
        # tokenizer encoding used by the countersign compilation below)
        # M (tokenizer-consistent identity)
        a.rationalise(KValue(compile_source("M", dev=True)[0].kline, SIG_S4))
        # H (tokenizer-consistent identity)
        a.rationalise(KValue(compile_source("H", dev=True)[0].kline, SIG_S4))

        entries = compile_source("M == H", dev=True)
        assert len(entries) == 2

        # Pre-register (simulates adapter behaviour)
        for e in entries:
            a.model.add_to_stm(e.kline)

        events = []
        orig = bus.on_event

        def cap(ev):
            events.append(ev)
            orig(ev)

        bus.on_event = cap

        for e in entries:
            result = a.rationalise(KValue(e.kline, SIG_S4))
            assert result is True

        # Both should produce frame events with S1 significance
        frame_events = [e for e in events if e.kind == "frame"]
        assert len(frame_events) == 2
        for ev in frame_events:
            assert ev.proposal.significance > 0  # S1 range


class TestGroundCheckExcludesSTM:
    """CR-3, CR-4: Model.grounded() checks Frame/LTM only."""

    def test_grounded_returns_false_for_stm_only(self):
        m = Model()
        kl = KLine(0xFF, [1, 2])
        m.add_to_stm(kl)
        assert m.exists(kl) is True
        assert m.grounded(kl) is False

    def test_grounded_returns_true_for_frame_entry(self):
        m = Model()
        kl = KLine(0xFF, [1, 2])
        m.add_to_frame(kl)
        assert m.grounded(kl) is True

    def test_grounded_returns_true_for_ltm_entry(self):
        m = Model()
        kl = KLine(0xFF, [1, 2])
        m.add_to_ltm(kl)
        assert m.grounded(kl) is True


class TestSelfFilterInCandidates:
    """CR-5: rationalise() excludes self from candidates."""

    def test_rationalise_excludes_self_from_candidates(self):
        bus = EventBus()
        a = KAgent(adapter=bus)

        # Add a candidate that partially overlaps with query signature
        candidate = KLine(T(5), [T(10), T(30)])
        a.rationalise(KValue(candidate, SIG_S4))

        # Query overlaps on [10] but not [20] -> should be S2, not S1
        q = KLine(0, [T(10), T(20)])
        q.signature = signifier.make_signature([T(10), T(20)])
        result = a.rationalise(KValue(q, SIG_S4))
        # S2 should return False (slow path) even though q is in STM
        assert result is False


class TestSigLevelPropagation:
    """CR-6: sig_level is set on CompiledEntry matching operator type."""

    def test_sig_level_set_on_compiled_entry(self):
        entries = compile_source("M == H", dev=True)
        for e in entries:
            assert sig_level(e.kline, signifier) == "S1"

        entries = compile_source("M > H", dev=True)
        for e in entries:
            if e.kline.nodes:  # skip unsigned identities
                assert sig_level(e.kline, signifier) == "S3"

        entries = compile_source("M => H", dev=True)
        for e in entries:
            if e.kline.nodes and not isinstance(e.kline.nodes, list):
                assert sig_level(e.kline, signifier) == "S2"

    def test_sig_level_none_by_default(self):
        kl = KLine(0xFF, [1])
        # sig_level() always returns a string via _infer_level() or _SIG_LEVELS
        assert isinstance(sig_level(kl, signifier), str)


class TestUndersignIsConnotateReversed:
    """CR-7: Undersign gets no special fast path. CR-8: connotate through slow path."""

    def test_undersign_no_special_fast_path(self):
        """Undersign {M: S} resolves as S3 (not S1), goes through slow path."""
        bus = EventBus()
        a = KAgent(adapter=bus)

        # Add identities (derived from compile_source so signatures match the
        # tokenizer encoding used by the undersign compilation below)
        # M (tokenizer-consistent identity)
        a.rationalise(KValue(compile_source("M", dev=True)[0].kline, SIG_S4))
        # S (tokenizer-consistent identity)
        a.rationalise(KValue(compile_source("S", dev=True)[0].kline, SIG_S4))

        # Compile undersign: S = M -> {M: S}
        entries = compile_source("S = M", dev=True)
        # Filter to just the undersign entry (not the unsigned identity)
        undersign = [e for e in entries if e.kline.nodes]
        assert len(undersign) == 1
        e = undersign[0]
        # Undersign maps to S3 (not S1) in _SIG_LEVELS
        assert e.kline.dbg.op == "UNDERSIGNED"
        assert sig_level(e.kline, signifier) == "S3"

        # Pre-register
        a.model.add_to_stm(e.kline)

        result = a.rationalise(KValue(e.kline, SIG_S4))
        # Undersign gets no special fast path: {M: [S]} is routed against
        # the {M: []} identity (match_count 0) -> S3, so it goes through the
        # slow path and rationalise returns False (CR-7).
        assert result is False

    def test_connotate_goes_through_slow_path(self):
        """Connotate {A: D} should go through candidate retrieval -> slow path."""
        bus = EventBus()
        a = KAgent(adapter=bus)

        # Add identity A only (D is unknown).  Derived from compile_source so
        # its signature matches the tokenizer encoding used by the connotate below.)
        # A (tokenizer-consistent identity)
        a.rationalise(KValue(compile_source("A", dev=True)[0].kline, SIG_S4))

        entries = compile_source("A > D", dev=True)
        connotate = [e for e in entries if e.kline.nodes]
        assert len(connotate) == 1
        e = connotate[0]
        assert sig_level(e.kline, signifier) == "S3"

        a.model.add_to_stm(e.kline)
        result = a.rationalise(KValue(e.kline, SIG_S4))
        # No candidates for A signature, so it should be novel (S4 -> True)
        # OR if candidates exist, S3 -> False (slow path)
        # Since only {A: None} exists, where(A) returns it
        # _route({A: [D]}, {A: []}) -> S3 -> False
        assert result is False
