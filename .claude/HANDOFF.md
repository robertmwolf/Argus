# Claude Code Session Start Prompt

Copy and paste this prompt as your FIRST message when you open
Claude Code in the Argus/ project directory.

---

## PASTE THIS INTO CLAUDE CODE:

```
Read CLAUDE.md, then read agent_docs/assistant_guide.md before doing anything else.

Once you have read those files, summarise:
1. What the system does in one sentence
2. What the current phase status is
3. What the next pending work item is

Then wait for my instruction before writing any code.
```

---

## Context (as of 2026-05-14)

- All implementation phases (0–8) are complete and 325 tests pass.
- DINOv3 ViT-B backbone (Phase C²) is merged to `main`.
  - Frozen ViT-B, full merged dataset, 4 epochs: **mAP@0.5=0.74** on test.json
  - Beats Co-DINO Swin-T (0.19) by +0.55 mAP@0.5
- Phase D (ViT-L, full dataset, 50 epochs, RTX 5070 Ti) is the only pending item.
  - See `agent_docs/Training_Handoff.md` for the handoff procedure.
- A structural refactor plan is documented in `agent_docs/refactor_plan.md`
  (rename `src/` → `classical/`, `agent_docs/` → `docs/`, `models/` → `configs/`).
  Not yet executed — wait until Phase D results are in hand.

## Useful session openers

### Continue Phase D prep / review handoff docs
```
Read agent_docs/Training_Handoff.md.
Summarise the Phase D DINOv3 ViT-L training procedure in bullet points.
Flag anything that looks stale or missing.
```

### Work on a specific bug or feature
```
Read agent_docs/assistant_guide.md.
I want to work on: <describe the task>
Plan the change before writing any code and wait for my approval.
```

### Run the structural refactor
```
Read agent_docs/refactor_plan.md.
Implement Issues 1–3 (directory renames) as a single coordinated commit.
Plan first, then wait for my approval before making any changes.
```

---

## Tips

- **Plan first, always.** Ask Claude to plan before coding and end the prompt
  with "wait for my approval before writing any code."
- **Run pytest after each module.** `conda activate satid && pytest tests/ -v`
- **One phase at a time.** Don't ask it to build multiple phases at once.
- **If context gets long,** start a new Claude Code session and re-read
  CLAUDE.md + the relevant section of assistant_guide.md.
