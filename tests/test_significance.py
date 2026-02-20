import pytest
from kalvin.model import (
    KLine,
    Model,
)
from kalvin.significance import (
    Significance,
    S1_BIT,
    S1_PCT_SHIFT,
    S2_SHIFT,
    S3_SHIFT,
    S4_VALUE,
    has_s1,
    get_s1_percentage,
    get_s2,
    get_s2_s1_percentage,
    get_s2_s2_percentage,
    build_s1,
    build_s2,
    get_s3,
    get_s3_s1_percentage,
    get_s3_s2_percentage,
    get_s3_gen_percentage,
    build_s3,
    calculate_significance,
)


class TestSignificanceHelpers:
    def test_build_s1_100_percent(self):
        """S1 at 100% sets S1 bit with max percentage (127)."""
        sig = build_s1(100)
        assert has_s1(sig) is True
        assert get_s1_percentage(sig) == 127

    def test_build_s1_50_percent(self):
        """S1 at 50% sets S1 bit with 63 in percentage bits."""
        sig = build_s1(50)
        assert has_s1(sig) is True
        assert get_s1_percentage(sig) == 63

    def test_build_s1_0_percent(self):
        """S1 at 0% still sets S1 bit with 0 percentage."""
        sig = build_s1(0)
        assert has_s1(sig) is True
        assert get_s1_percentage(sig) == 0

    def test_build_s1_clamps_negative(self):
        """Negative percentage is clamped to 0."""
        sig = build_s1(-10)
        assert has_s1(sig) is True
        assert get_s1_percentage(sig) == 0

    def test_build_s1_clamps_over_100(self):
        """Percentage over 100 is clamped to 100."""
        sig = build_s1(150)
        assert has_s1(sig) is True
        assert get_s1_percentage(sig) == 127

    def test_build_s1_default(self):
        """S1 with no percentage defaults to 100%."""
        sig = build_s1()
        assert has_s1(sig) is True
        assert get_s1_percentage(sig) == 127

    def test_s1_bit_value(self):
        """S1 bit is at position 56."""
        sig = build_s1()
        assert sig == S1_BIT | (127 << S1_PCT_SHIFT)

    def test_build_s2_full(self):
        """S2 with both percentages gives correct values."""
        sig = build_s2(50, 50)
        assert get_s2_s1_percentage(sig) == 127
        assert get_s2_s2_percentage(sig) == 127

    def test_build_s2_zero_s1(self):
        """S2 with zero S1%."""
        sig = build_s2(0, 100)
        assert get_s2_s1_percentage(sig) == 0
        assert get_s2_s2_percentage(sig) == 255

    def test_build_s2_zero_s2(self):
        """S2 with zero S2%."""
        sig = build_s2(100, 0)
        assert get_s2_s1_percentage(sig) == 255
        assert get_s2_s2_percentage(sig) == 0

    def test_get_s2_returns_combined(self):
        """get_s2 returns the full 16-bit S2 value."""
        sig = build_s2(100, 100)
        assert get_s2(sig) == 0xFFFF

    def test_s4_value_is_zero(self):
        """S4 (no significance) is 0."""
        assert S4_VALUE == 0


class TestCalculateSignificance:
    def test_s1_exact_match(self):
        """Exact node match returns S1."""
        query = KLine(s_key=0x1000, nodes=[0x100, 0x200])
        model_kline = KLine(s_key=0x2000, nodes=[0x100, 0x200])
        m = Model([query, model_kline])

        sig = calculate_significance(m,query, model_kline)
        assert has_s1(sig) is True

    def test_s1_prefix_match_query_shorter(self):
        """Query prefix matches model (query shorter)."""
        query = KLine(s_key=0x1000, nodes=[0x100])
        model_kline = KLine(s_key=0x2000, nodes=[0x100, 0x200])
        m = Model([query, model_kline])

        sig = calculate_significance(m,query, model_kline)
        assert has_s1(sig) is True  # All prefix nodes match

    def test_s1_prefix_match_query_longer(self):
        """Query prefix matches model (query longer)."""
        query = KLine(s_key=0x1000, nodes=[0x100, 0x200])
        model_kline = KLine(s_key=0x2000, nodes=[0x100])
        m = Model([query, model_kline])

        sig = calculate_significance(m,query, model_kline)
        assert has_s1(sig) is True  # All prefix nodes match

    def test_s1_empty_both(self):
        """Both empty nodes returns S1."""
        query = KLine(s_key=0x1000, nodes=[])
        model_kline = KLine(s_key=0x2000, nodes=[])
        m = Model([query, model_kline])

        sig = calculate_significance(m,query, model_kline)
        assert has_s1(sig) is True

    def test_s4_query_empty_model_not(self):
        """Query empty, model not empty returns S4."""
        query = KLine(s_key=0x1000, nodes=[])
        model_kline = KLine(s_key=0x2000, nodes=[0x100])
        m = Model([query, model_kline])

        sig = calculate_significance(m,query, model_kline)
        assert sig == S4_VALUE

    def test_s4_model_empty_query_not(self):
        """Model empty, query not empty returns S4."""
        query = KLine(s_key=0x1000, nodes=[0x100])
        model_kline = KLine(s_key=0x2000, nodes=[])
        m = Model([query, model_kline])

        sig = calculate_significance(m,query, model_kline)
        assert sig == S4_VALUE

    def test_s2_partial_match(self):
        """Partial positional match returns S2."""
        query = KLine(s_key=0x1000, nodes=[0x100, 0x200])
        model_kline = KLine(s_key=0x2000, nodes=[0x100, 0x300])
        m = Model([query, model_kline])

        sig = calculate_significance(m,query, model_kline)
        assert has_s1(sig) is False  # Not S1
        assert get_s2_s1_percentage(sig) == 127  # 50% positional match

    def test_s2_with_non_positional_match(self):
        """S2 includes non-positional matches."""
        query = KLine(s_key=0x1000, nodes=[0x100, 0x200])
        model_kline = KLine(s_key=0x2000, nodes=[0x100, 0x300, 0x200])  # 0x200 at pos 2
        m = Model([query, model_kline])

        sig = calculate_significance(m,query, model_kline)
        assert has_s1(sig) is False  # Not S1
        assert get_s2_s1_percentage(sig) == 127  # 50% positional
        assert get_s2_s2_percentage(sig) == 127  # 50% non-positional

    def test_s4_no_match(self):
        """No matching nodes returns S4."""
        query = KLine(s_key=0x1000, nodes=[0x100])
        model_kline = KLine(s_key=0x2000, nodes=[0x200])
        m = Model([query, model_kline])

        sig = calculate_significance(m,query, model_kline)
        assert sig == S4_VALUE


class TestSignificanceComparison:
    def test_s1_greater_than_s2(self):
        """S1 is more significant than S2."""
        s1_sig = build_s1(50)
        s2_sig = build_s2(100, 100)
        assert s1_sig > s2_sig

    def test_s1_100_greater_than_s1_50(self):
        """Higher S1% is more significant."""
        sig_high = build_s1(100)
        sig_low = build_s1(50)
        assert sig_high > sig_low

    def test_s2_greater_than_s4(self):
        """S2 is more significant than S4."""
        s2_sig = build_s2(1, 1)
        assert s2_sig > S4_VALUE

    def test_s2_higher_s1_pct_more_significant(self):
        """S2 with higher S1% is more significant."""
        sig_high = build_s2(100, 0)
        sig_low = build_s2(50, 0)
        assert sig_high > sig_low

    def test_s2_higher_s2_pct_more_significant(self):
        """S2 with higher S2% is more significant (same S1%)."""
        sig_high = build_s2(50, 100)
        sig_low = build_s2(50, 50)
        assert sig_high > sig_low

    def test_s1_s2_s4_ordering(self):
        """Full ordering: S1 > S2 > S4."""
        s1 = build_s1(100)
        s2 = build_s2(100, 100)
        s4 = S4_VALUE
        assert s1 > s2 > s4

    def test_calculated_significance_ordering(self):
        """Real calculated significances maintain ordering."""
        m = Model()

        # S1: exact match
        q = KLine(s_key=0x1000, nodes=[0x100, 0x200])
        t1 = KLine(s_key=0x2000, nodes=[0x100, 0x200])
        m.add(q)
        m.add(t1)
        sig_s1 = calculate_significance(m,q, t1)

        # S2: partial match
        t2 = KLine(s_key=0x3000, nodes=[0x100, 0x300])
        m.add(t2)
        sig_s2 = calculate_significance(m,q, t2)

        # S4: no match
        t3 = KLine(s_key=0x4000, nodes=[0x999])
        m.add(t3)
        sig_s4 = calculate_significance(m,q, t3)

        assert sig_s1 > sig_s2 > sig_s4


class TestSignificanceHelpersS3:
    def test_build_s3_full(self):
        """S3 with all percentages gives correct values."""
        sig = build_s3(100, 100, 100)
        assert get_s3_s1_percentage(sig) == 255
        assert get_s3_s2_percentage(sig) == 255
        assert get_s3_gen_percentage(sig) == 255

    def test_build_s3_partial(self):
        """S3 with partial percentages."""
        sig = build_s3(50, 50, 50)
        assert get_s3_s1_percentage(sig) == 127
        assert get_s3_s2_percentage(sig) == 127
        assert get_s3_gen_percentage(sig) == 127

    def test_build_s3_zero(self):
        """S3 with zero percentages."""
        sig = build_s3(0, 0, 0)
        assert get_s3(sig) == 0

    def test_get_s3_returns_combined(self):
        """get_s3 returns the full 24-bit S3 value."""
        sig = build_s3(100, 100, 100)
        assert get_s3(sig) == 0xFFFFFF


class TestCalculateSignificanceS3:
    def test_s3_unordered_match(self):
        """S3 when nodes match but at different positions (no positional overlap)."""
        # Query: [a, b], Model: [b, c, a] - a and b exist but not at same positions
        query = KLine(s_key=0x1000, nodes=[0x100, 0x200])
        model_kline = KLine(s_key=0x2000, nodes=[0x300, 0x100])  # 0x100 at different position
        m = Model([query, model_kline])

        sig = calculate_significance(m,query, model_kline)
        assert has_s1(sig) is False  # Not S1
        assert get_s2(sig) == 0  # Not S2 (no positional matches)
        assert get_s3_s1_percentage(sig) > 0  # Has unordered S1 matches

    def test_s3_reversed_nodes(self):
        """S3 when nodes are in reverse order."""
        query = KLine(s_key=0x1000, nodes=[0x100, 0x200])
        model_kline = KLine(s_key=0x2000, nodes=[0x200, 0x100])  # Reversed
        m = Model([query, model_kline])

        sig = calculate_significance(m,query, model_kline)
        assert get_s3_s1_percentage(sig) == 255  # 100% unordered match

    def test_s3_generational_match(self):
        """S3 generational match through child nodes."""
        # K1 -> N1 -> N3, K2 has N3 directly
        n3 = KLine(s_key=0x0030, nodes=[])
        n1 = KLine(s_key=0x0010, nodes=[0x0030])  # N1 has child N3
        k1 = KLine(s_key=0x1000, nodes=[0x0010])  # K1 has child N1
        k2 = KLine(s_key=0x2000, nodes=[0x0020, 0x0030])  # K2 has N2 and N3
        m = Model([n3, n1, k1, k2])

        sig = calculate_significance(m,k1, k2)
        # K1's node N1 has child N3 which matches K2's node N3
        assert get_s3_s2_percentage(sig) > 0  # Child match

    def test_s3_no_match_returns_s4(self):
        """No unordered or generational match returns S4."""
        query = KLine(s_key=0x1000, nodes=[0x100])
        model_kline = KLine(s_key=0x2000, nodes=[0x200])
        m = Model([query, model_kline])

        sig = calculate_significance(m,query, model_kline)
        assert sig == S4_VALUE


class TestSignificanceComparisonS3:
    def test_s2_greater_than_s3(self):
        """S2 is more significant than S3."""
        s2_sig = build_s2(1, 1)
        s3_sig = build_s3(100, 100, 100)
        assert s2_sig > s3_sig

    def test_s3_greater_than_s4(self):
        """S3 is more significant than S4."""
        s3_sig = build_s3(1, 0, 0)
        assert s3_sig > S4_VALUE

    def test_s3_higher_s1_pct_more_significant(self):
        """S3 with higher S1% is more significant."""
        sig_high = build_s3(100, 0, 0)
        sig_low = build_s3(50, 0, 0)
        assert sig_high > sig_low

    def test_s3_higher_gen_pct_more_significant(self):
        """S3 with higher gen% is more significant."""
        sig_high = build_s3(0, 0, 100)
        sig_low = build_s3(0, 0, 50)
        assert sig_high > sig_low

    def test_full_ordering(self):
        """Full ordering: S1 > S2 > S3 > S4."""
        s1 = build_s1(100)
        s2 = build_s2(100, 100)
        s3 = build_s3(100, 100, 100)
        s4 = S4_VALUE
        assert s1 > s2 > s3 > s4

    def test_calculated_full_ordering(self):
        """Real calculated significances maintain full ordering."""
        # Build a model with various KLines for testing
        n3 = KLine(s_key=0x0030, nodes=[])

        q = KLine(s_key=0x1000, nodes=[0x100, 0x200])  # Query

        # S1: exact match
        t1 = KLine(s_key=0x2000, nodes=[0x100, 0x200])

        # S2: partial positional match
        t2 = KLine(s_key=0x3000, nodes=[0x100, 0x300])

        # S3: only unordered match (reversed)
        t3 = KLine(s_key=0x4000, nodes=[0x200, 0x100])

        # S4: no match
        t4 = KLine(s_key=0x5000, nodes=[0x999])

        m = Model([n3, q, t1, t2, t3, t4])

        sig_s1 = calculate_significance(m,q, t1)
        sig_s2 = calculate_significance(m,q, t2)
        sig_s3 = calculate_significance(m,q, t3)
        sig_s4 = calculate_significance(m,q, t4)

        assert sig_s1 > sig_s2 > sig_s3 > sig_s4
