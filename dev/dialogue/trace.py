"""The trace — a run's optimal path, read from the Cogitator after it
ends. The model searches; this module narrates: the opening pairing
and, for each side it rests on, the one chain that carried it — the
child's resolving row, the exchange pair, or the source of the one
frame that covers the region. One row per reentry — paths stay
orthogonal. In generation order."""

from rationalisation.model import Cogitator, KNode, Row, mu


def _cover_frame(cog: Cogitator, x: KNode, chain: set[int]):
    """The smallest frame covering the region that a proper ancestor row
    produced — or a joint/exchange frame. The reentry's own rows and its
    siblings' never cover."""
    best = None
    for f, fr in zip(cog.frames, cog.frame_re):
        if fr is not None and id(fr) not in chain:
            continue
        if x & f == x and (best is None or mu(f) < mu(best)):
            best = f
    return best


def trace(cog: Cogitator) -> list[Row]:
    """The optimal path."""
    if cog.claim is None or cog.goal_row is None:
        return []
    rindex: dict[int, int] = {id(r): i for i, r in enumerate(cog.rows)}  # identity — Row eq recurses
    keep: dict[int, Row] = {}
    tabled: dict[int, int] = {}  # reentry id -> the row id holding its slot
    via_frame: set[int] = set()  # rows kept as frame sources — they pull no pushers
    # rows reached through exchange causality table alongside the slot —
    # the S3 path and the shortcut that resolved first are two orthogonal
    # paths of one reentry; no row serves two reentries.
    feeds: set[int] = set()
    stack: list[Row] = [cog.goal_row]
    while stack:
        row = stack.pop()
        if id(row) in keep:
            continue
        re_ = row.re
        if re_ is not None and tabled.get(id(re_), id(row)) != id(row) and id(row) not in feeds:
            continue  # this entry already carries its row — a feed is a second path
        keep[id(row)] = row
        if re_ is not None and id(row) not in feeds:
            tabled[id(re_)] = id(row)  # feeds table alongside, never claiming the slot
        if (
            re_ is not None
            and re_.parent is not None
            and id(row) not in via_frame
        ):
            stack.append(re_.parent)  # the row that pushed this reentry is on the path
        for side, region in (("u", row.U), ("o", row.O)):
            if not region:
                continue
            child = next((ch for ch in row.children if ch.side == side), None)
            if child is not None and child.resolved_by is not None:
                # the earliest-created resolving row carries the slot —
                # closers and scaffolding pair first, so the reductive
                # chain wins over the mirror that happened to close first
                slot = next((r for r in child.rows if r.resolved), None) or child.resolved_by
                stack.append(slot)
                continue
            if child is not None and child.stalled is not None and child.stalled.resolved:
                for sr in cog.frame_src.get(int(child.kline.sig), [])[:2]:  # the exchange pair
                    stack.append(sr)
                    if sr.re is not None and sr.re.parent is not None:
                        feeds.add(id(sr.re.parent))  # the row that fed the stall — a second path
                        stack.append(sr.re.parent)
                continue
            chain = cog._chain(row.re)
            best = None
            for i, (f, fr) in enumerate(zip(cog.frames, cog.frame_re)):
                if fr is not None and id(fr) not in chain:
                    continue  # only ancestor and joint frames settle a side
                if region & f == region and (best is None or (mu(f), i) < best):
                    best = (mu(f), i, f)
            if best is not None:
                srcs = cog.frame_src.get(int(best[2]), [])
                if srcs:
                    # earliest-created source — consistent with the slot rule
                    src = srcs[0]
                    for s in srcs[1:]:
                        if rindex[id(s)] < rindex[id(src)]:
                            src = s
                    via_frame.add(id(src))
                    stack.append(src)
    # repair: every tabled row traces back to the root through tabled
    # pushers — a row whose reentry no tabled row claims does not trace
    # back, and is dropped; the drops cascade
    for _ in range(len(keep) + 1):
        claimed: set[int] = set()
        for r in keep.values():
            for ch in r.children:
                if (ch.stalled is not None and id(ch.stalled) in keep) or any(
                    id(x) in keep for x in ch.rows
                ):
                    claimed.add(id(ch))
        floating = next(
            (
                r
                for r in keep.values()
                if r is not cog.goal_row
                and r.re is not None
                and id(r.re) not in claimed
            ),
            None,
        )
        if floating is None:
            break
        del keep[id(floating)]  # does not trace back to the root — drop
    # the frames each kept row leans on — its fit, and any side not
    # carried by a tabled child — each with the row that established it
    tset = set(keep)
    for row in keep.values():
        sup: list[tuple[int, Row]] = []
        chain = cog._chain(row.re)
        own_id = None  # frames own their rows now — the chain excludes siblings

        def lean(x: KNode) -> None:
            f = _cover_frame(cog, x, chain)
            if f is None:
                rf = next((v for v, _ in cog.rframes if x & v == x), None)  # a resolution's own record
                if rf is not None:
                    f = rf
            if f is None:
                return
            srcs: list[Row] = []
            for s in cog.frame_src.get(int(f), []):
                if not any(id(s) == id(t) for t in srcs):
                    srcs.append(s)
            for s in (r for v, r in cog.rframes if v == f):
                if not any(id(s) == id(t) for t in srcs):
                    srcs.append(s)
            if not srcs:
                return
            onpath = [s for s in srcs if id(s) in tset]
            pool = onpath or srcs  # tabled rows first
            if pool:
                pick = pool[0]
                for s in pool[1:]:  # earliest created — identity index, rows never compare
                    if rindex[id(s)] < rindex[id(pick)]:
                        pick = s
                sup.append((int(f), pick))

        if row.F:
            lean(row.F)
        root = row is cog.goal_row
        for side, region in (("u", row.U), ("o", row.O)):
            if not region:
                continue
            child = next((ch for ch in row.children if ch.side == side), None)
            carried = not root and child is not None and (
                (child.stalled is not None and id(child.stalled) in tset)
                or any(id(x) in tset for x in child.rows)
            )
            # the claim's own two regions always lean — the grounded frame
            if not carried:
                lean(region)
        own = int(row.frame) if row.frame else None
        seen: set[tuple[int, int]] = set()
        row.supports = tuple(
            p
            for p in sup
            if p[0] != own and not (p[0], id(p[1])) in seen and not seen.add((p[0], id(p[1])))
        )
    return [r for r in cog.rows if id(r) in keep]
