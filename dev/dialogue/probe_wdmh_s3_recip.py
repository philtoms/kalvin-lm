import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")
from collections import deque
from kalvin.derivation import Derivation, KLine
from kalvin.kvalue import KValue
from kalvin.significance import SIG_S1

def slot_walk_s3(self, slot, end_mask=None):
    if end_mask is None:
        end_mask = self.excess()
    queue = deque([([slot], frozenset(), 0, [])])
    seen = {(int(slot),)}
    expanded = 0
    while queue:
        if expanded >= self.max_walk_states:
            return None
        expanded += 1
        nodes, used, edges, path = queue.popleft()
        if edges and self.signifier.signifies(
            int(self.signifier.signature_of(nodes)), end_mask
        ):
            return nodes, edges, path
        if edges >= self.max_walk_edges:
            continue
        for k in self.memory:
            if not self.usable(k):
                continue
            key = (int(k.signature), tuple(int(n) for n in k.nodes))
            if key in used:
                continue
            if self.occurs_rev(k, nodes):
                new = self.replace_rev(k, nodes)
                t = tuple(sorted(int(x) for x in new))
                if t in seen:
                    continue
                seen.add(t)
                queue.append((new, used | {key}, edges + 1, path + [("reverse", k)]))
    return None
Derivation.slot_walk = slot_walk_s3

def walk_b_slot_headed(self) -> bool:
    for slot in self.goal.nodes:
        if not self.signifier.signifies(slot, self.excess()):
            continue
        walk = self.slot_walk(slot, end_mask=self.content())
        if walk is None:
            continue
        end, edges, path = walk
        end, edges, _ = self.refine(end, edges, path)
        if not any(n in self.nodes for n in end):
            continue
        composed = KLine(slot, end, acq_depth=edges)
        if self._ground_composed(composed):
            self.composed.append(composed)
            return True
    return False
Derivation._walk_b = walk_b_slot_headed

def nm(v): return getattr(v, 'label', '') or hex(int(v))
orig_run = Derivation.run
def run(self):
    is_wdmh = getattr(self.queued.signature, 'label', '') == 'WDMH'
    r = orig_run(self)
    if is_wdmh:
        g = f"{nm(self.goal.signature)}:[{','.join(nm(n) for n in self.goal.nodes)}]"
        print(f"derivation goal={g} -> {r.ending} j1={r.j1:.2f} "
              f"trace={['[' + ', '.join(nm(n) for n in s) + ']' for s in r.trace]} "
              f"composed={[f'{nm(k.signature)}:[{chr(44).join(nm(n) for n in k.nodes)}]' for k in r.composed]}")
    return r
Derivation.run = run

from dev.dialogue.harness import load_engine
from kalvin.bpe_tokenizer import BPETokenizer
from ks.compiler import compile_source
tok = BPETokenizer()
h = load_engine('data/dialogue/mhall.json', tok)
src = open('data/scripts/wdmh.ks').read()
entries = compile_source(src, tokenizer=tok, signifier=h.signifier, dev=True,
                         word_bits=h.word_bits, known_words=h.known_words)
h.engine.rationalise(entries)
den = next(k for b in h.state.frame.values() for k in b
           if getattr(k.signature, 'label', '') == 'what' and getattr(k.nodes[0], 'label', '') == 'Object')
h.engine.rationalise([KValue(KLine(den.nodes[0], [den.signature]), SIG_S1)])
h.run(src)
print("held WDMH:", [f"{nm(k.signature)}:[{', '.join(nm(n) for n in k.nodes)}]"
                     for b in h.state.frame.values() for k in b
                     if getattr(k.signature, 'label', '') == 'WDMH' and k.nodes])

print("\nframe WDMH-bucket keys:")
from kalvin.kline import is_ask
for sig, bucket in h.state.frame.items():
    if 'WDMH' in (getattr(sig, 'label', '') or '') or (int(sig) & ~0x80000000) in (0x803000066f0f,):
        for k in bucket:
            print(f"  key ask={is_ask(sig)} [{', '.join(nm(n) for n in k.nodes)}]")
