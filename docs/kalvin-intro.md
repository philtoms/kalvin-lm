# Kalvin — A Rationalising System

Kalvin is a rationalising system that accepts, thinks, and talks in klines. It receives kline structures — compiled from KScript by an agent — and attempts to understand each one in terms of what it already knows. Understanding, for Kalvin, is not a matter of semantics. It is a matter of shape.

A kline is a node-like structure. Klines combine to describe the world in which Kalvin operates. There are only two kinds: **identities** and **relationships**. An identity kline is a node with no further links — it simply exists. A relationship kline is a node that links to other klines. Every relationship kline has a **query** (its head) and a **value** (the remaining nodes). When Kalvin receives a relationship kline, it examines the query and asks: do I know what this is?

Kalvin's world is built entirely from these two kinds of structures. It has no access to the scripts that produced them, no concept of the characters or words they represent, and no sense of the linguistic or semantic relationships that might be obvious to a human reader. Where we see meaning in the idea of a subject or a verb, Kalvin sees meaning in how one kline connects to another. Where we might say "M stands for Mary," Kalvin sees only that M is linked to S, and S is linked to SVO, and SVO is linked to MHALL. This is not a limitation. It is the point. Meaning for Kalvin is found in shape.

When Kalvin receives a kline, it **rationalises** it: it searches its existing knowledge for klines that share structure with the new arrival, measures how closely they relate, and produces a response — a freshly generated kline that represents its best understanding. Attached to this response is a single numerical indicator called **significance**, which expresses how much of the new kline Kalvin was able to relate to what it already knows. The higher the significance, the more certainty Kalvin has in the relationship it has constructed.

Significance falls on a spectrum. At one end, every node in the new kline resolves to grounded knowledge — Kalvin fully understands what it has received. At the other end, nothing connects at all — the kline is entirely novel. Between these two extremes lie degrees of partial understanding: some nodes match, some don't, or the connections are associative rather than direct. In every case, significance is not a label Kalvin assigns after the fact. It is a direct consequence of how the new kline fits — or doesn't — into the shape of Kalvin's existing knowledge.

Kalvin rationalises every kline it receives in exactly the same way, regardless of context. There is no training mode, no operational flag, no special behaviour triggered by circumstance. A kline arrives, Kalvin searches its knowledge, computes significance, and responds. Whether that kline came from a deliberate training exercise or a live query, Kalvin's process is identical.

What differs is the **harness** — the code that manages the flow of klines and responses between the agent and Kalvin. A training harness compiles scripts and uses significance to decide what to teach next. An operational harness presents live queries and uses significance to decide how to act on the response. Different harnesses for different jobs; Kalvin is unaware of the distinction. It simply rationalises what it receives and reports how well it understood.

From Kalvin's perspective, the world is a dialog. Klines arrive, one after another. Some are familiar — Kalvin recognises the shape immediately, responds with high significance, and the exchange is straightforward. Others are partially familiar — Kalvin can trace some of the new kline's nodes through its existing knowledge, but not all. It responds with what it understands and attaches a significance that honestly reflects its confidence. And sometimes a kline arrives that Kalvin cannot connect to anything at all. It says so — low significance, no match, I don't understand. This is not a failure. It is an honest answer, and it is often the most useful one. A low-significance response tells the other side of the dialog exactly where Kalvin's knowledge ends, which is precisely where the next kline should begin.

Consider the exchange:

```
  (What did mary have?)
  WDMH => M H W
```

Kalvin receives this as a compiled kline. It recognises M and H — they connect to S and V, which connect to SVO, which connects to MHALL. Kalvin can trace a path from M and H back to MHALL and responds accordingly. But W does not connect to anything. The gap is visible in the significance: the response is partially grounded but not fully. Kalvin says what it knows and indicates where it doesn't. The next kline to arrive might connect W to something Kalvin already knows — perhaps W to ALL, or W to O. If so, Kalvin integrates the new relationship, the gap closes, and a similar exchange in future would route directly to high significance.

When the dialog pauses, Kalvin continues to think. Any kline that landed at partial significance represents material Kalvin has seen but not fully internalised. During these quiet periods, Kalvin revisits its partial knowledge — retracing paths, discovering connections that were not apparent before, strengthening its own understanding. This study cannot fully confirm a kline — full understanding arrives when the dialog resumes and a new kline routes cleanly to high significance — but it prepares. The next exchange may be immediately recognisable, requiring no further scaffolding, because Kalvin has done the work of connecting what it already knows.

This is how Kalvin grows: not by adjusting weights or minimising error, but by building a knowledge graph one kline at a time. Each exchange adds structure. Each pause strengthens it. Over time, the graph becomes rich enough that new klines fit into familiar shapes, and Kalvin's responses carry the confidence of a system that has seen enough of the world to understand what it is being shown.
