"""Tests for the dialogue runner — resolution + end-to-end session.

Covers the symbolic resolver (legend role → kline), the dialogue → ResponseTable
mapping (including the shared-BPE-constituent edge case), and a full end-to-end
session driven through the real MessageBus + KAgentAdapter + StubKAgent.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import dialogue_runner as dr  # noqa: E402

from kalvin.expand import SIG_S1, SIG_S2, SIG_S3, SIG_S4  # noqa: E402
from kalvin.kline import KLine  # noqa: E402
from kalvin.kvalue import KValue  # noqa: E402
from kalvin.nlp_tokenizer import NLPTokenizer  # noqa: E402
from kalvin.signifier import NLPSignifier  # noqa: E402

DIALOGUE_PATH = _SCRIPTS / "dialogue-mhall.json"


@pytest.fixture(scope="module")
def dialogue() -> dict:
    return json.loads(DIALOGUE_PATH.read_text())


@pytest.fixture(scope="module")
def resolved(dialogue):
    return dr.resolve_dialogue(dialogue)


# ── Role resolution ──────────────────────────────────────────────────────


class TestRoleResolution:
    """Each symbolic legend role resolves to a concrete kline."""

    def test_primary_resolves_to_countersigned_entry(self, dialogue):
        r = dr.RoleResolver(dialogue["legend"], dialogue["source"], NLPTokenizer(), NLPSignifier())
        kl = r.kline_of("primary")
        # The primary carries MHALL's signature (its signature_role) and one node (SVO).
        assert kl.signature == r.sig_of("MHALL")
        assert len(kl.nodes) == 1

    def test_multitoken_word_resolves_to_its_subword_canon(self, dialogue):
        r = dr.RoleResolver(dialogue["legend"], dialogue["source"], NLPTokenizer(), NLPSignifier())
        # M (Mary) is a multi-token word → its kline IS the subword canon (same sig).
        assert r.kline_of("M") == r.kline_of("subword_Mary")

    def test_orphan_atom_is_synthesised(self, dialogue):
        r = dr.RoleResolver(dialogue["legend"], dialogue["source"], NLPTokenizer(), NLPSignifier())
        # "Mod" appears only as a relation operand, never emitted standalone →
        # synthesised as an identity kline (sig, []).
        kl = r.kline_of("Mo")
        assert kl.nodes == []
        assert kl.signature == r.sig_of("Mo")

    def test_shared_bpe_constituent_resolves_positionally(self, dialogue):
        r = dr.RoleResolver(dialogue["legend"], dialogue["source"], NLPTokenizer(), NLPSignifier())
        # little and lamb share their first BPE piece ('l'): sw_L1_1 == sw_L2_1.
        assert r.kline_of("sw_L1_1") == r.kline_of("sw_L2_1")
        # …but their second constituents differ.
        assert r.kline_of("sw_L1_2") != r.kline_of("sw_L2_2")

    def test_derived_sig_canonical_form_is_the_canon(self, dialogue):
        r = dr.RoleResolver(dialogue["legend"], dialogue["source"], NLPTokenizer(), NLPSignifier())
        # A DERIVED_SIG (MHALL) has no own kline; its structural form is the canon.
        assert r.kline_of("MHALL") == r.kline_of("canon_MHALL")

    def test_unresolved_role_raises(self, dialogue):
        r = dr.RoleResolver(dialogue["legend"], dialogue["source"], NLPTokenizer(), NLPSignifier())
        with pytest.raises(dr.ResolveError):
            r.kline_of("nonexistent_role")


# ── Dialogue → table ─────────────────────────────────────────────────────


class TestResolveDialogue:
    def test_submissions_are_one_kline_at_a_time_in_order(self, resolved, dialogue):
        # 27 trainer turns' klines, flattened to one-KValue-per-submission.
        assert len(resolved.submissions) == 27
        assert all(isinstance(kv.kline, KLine) for kv in resolved.submissions)

    def test_re_submission_keeps_one_row_per_unique_kline(self, resolved):
        # The shared 'l' piece (sw_L1_1/sw_L2_1) yields one row, not two.
        triggers = {row.trigger.kline for row in resolved.table}
        assert len(triggers) == len(resolved.table)  # no duplicate-trigger rows

    def test_expected_matches_table_emission_order(self, resolved):
        # expected is requests→grounds→countersigns per row (ST-7), in row order.
        rebuilt = [
            kv
            for row in resolved.table
            for kv in (*row.requests, *row.grounds, *row.countersigns)
        ]
        assert len(resolved.expected) == len(rebuilt)
        for got, exp in zip(resolved.expected, rebuilt):
            assert got.kline == exp.kline and got.significance == exp.significance

    def test_requests_are_s4_grounds_use_structural_band(self, resolved):
        for row in resolved.table:
            for req in row.requests:
                assert req.significance == SIG_S4
            for cs in row.countersigns:
                assert cs.significance == SIG_S1
        # grounds span S2/S3/S4 (canon / relation / atom)
        ground_sigs = {g.significance for row in resolved.table for g in row.grounds}
        assert ground_sigs <= {SIG_S2, SIG_S3, SIG_S4}
        assert SIG_S2 in ground_sigs and SIG_S4 in ground_sigs

    def test_turn_expected_aligned_with_trainer_turns(self, resolved):
        assert len(resolved.turn_expected) == len(resolved.trainer_turns)

    def test_turn_expected_flattens_to_expected(self, resolved):
        flat = [kv for turn in resolved.turn_expected for kv in turn]
        assert len(flat) == len(resolved.expected)
        for got, exp in zip(flat, resolved.expected):
            assert got.kline == exp.kline
            assert got.significance == exp.significance


# ── End-to-end session ───────────────────────────────────────────────────


class TestSession:
    def test_session_passes_against_real_harness(self, resolved):
        report = dr.run_session(resolved)
        assert report.ok, "session did not pass"
        assert report.primary_countersigned

    def test_every_expected_proposal_was_emitted_in_order(self, resolved):
        report = dr.run_session(resolved)
        assert len(report.captured) == len(report.expected)
        assert report.n_match == len(report.expected)

    def test_primary_countersign_is_the_final_emission(self, resolved):
        report = dr.run_session(resolved)
        last = report.captured[-1]
        assert last.kline == resolved.primary
        assert last.significance == SIG_S1

    def test_rows_fire_once_per_unique_trigger(self, resolved):
        report = dr.run_session(resolved)
        assert report.fired == len(resolved.table)

    def test_each_turn_reply_matches_table(self, resolved):
        # run_session verifies the kagent's reply against the table after every
        # paced turn; completing without raising proves each captured slice
        # equals the table's prescribed proposals, in order.
        report = dr.run_session(resolved)
        assert report.ok

    def test_mismatched_turn_expected_raises(self, resolved):
        bad = copy.deepcopy(resolved)
        kv0 = bad.turn_expected[0][0]
        bad.turn_expected[0][0] = KValue(kv0.kline, SIG_S1)  # wrong band
        with pytest.raises(dr.PacingError):
            dr.run_session(bad)

    def test_wrong_proposal_count_raises(self, resolved):
        bad = copy.deepcopy(resolved)
        bad.turn_expected[0] = bad.turn_expected[0][:1]  # expect one fewer
        with pytest.raises(dr.PacingError):
            dr.run_session(bad)
