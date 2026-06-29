# Archive Index

Append-only finding aid for deleted cascade artifacts. One row per artifact:
`artifact → where its conclusion migrated → recovery tag`. Git is the archive;
the repo is lean. To recover a file's pre-deletion form:

```bash
git show <tag>:<artifact>
```

| Artifact | Conclusion migrated to | Recovery tag |
|----------|-----------------------|-------------|
| `specs/reactive-delegation.md` | `specs/supervisor-decision.md` (SD-1…SD-21 re-own the surviving contracts; flag/budget/delegated-mode removals encoded as positive rules) | `docs-archive-2026-06-29` |
| `plans/impl/reactive-delegation.md` | Implemented — conclusion absorbed into `specs/supervisor-decision.md`; design rationale retained in code comments where code-alone-would-mislead | `docs-archive-2026-06-29` |
| `plans/impl/cascade-control.md` | Superseded by `plans/impl/realign-training-roles.md` (T4) — its reactive-round budget and silent-drop behaviour removed under `specs/supervisor-decision.md` SD-3 | `docs-archive-2026-06-29` |
