# Claude Code Handoff — Session Start Prompt

Copy and paste this prompt as your FIRST message when you open
Claude Code in the Argus/ project directory.

---

## PASTE THIS INTO CLAUDE CODE:

```
Read CLAUDE.md, then read agent_docs/architecture.md and
agent_docs/phase1_goals.md before doing anything else.

Once you have read those three files, tell me:
1. What the system does in one sentence
2. What Week 1 requires you to build
3. What the Week 1 success metric is

Then, before writing any code, give me a brief plan for
src/ingest/fits_parser.py:
- What dataclass(es) will you define and where
- What the main parse function signature will look like
- What the __main__ block will do
- What tests you will write

Wait for my approval of the plan before writing any code.
```

---

## Week-by-Week Prompts

After each week is done and tests are passing, use the
relevant prompt below to start the next week.

### Start Week 2
```
Week 1 is complete and pytest passes.
Read agent_docs/phase1_goals.md Week 2 section.
Plan src/detection/classical_detector.py before writing it.
Include: class/function signatures, preprocessing steps,
ASTRiDE integration approach, __main__ behavior, test plan.
Wait for approval before coding.
```

### Start Week 3
```
Week 2 is complete and pytest passes.
Read agent_docs/phase1_goals.md Week 3 section
and agent_docs/spacetrack.md.
Plan src/astrometry/plate_solver.py and
src/matching/spacetrack_query.py.
Show me signatures and caching design before coding.
```

### Start Week 4
```
Week 3 is complete and pytest passes.
Read agent_docs/phase1_goals.md Week 4 section.
Plan the four remaining modules:
  spatial_filter.py, propagator.py, matcher.py, scorer.py
and the end-to-end test.
Show me how data flows between them and the scoring formulas
you'll implement before coding.
```

### Record Baseline Metrics
```
All Week 4 modules are built and pytest passes.
Run test_end_to_end.py against the confirmed passes in
results/confirmed_passes.json.
Record all metrics to results/phase1_baseline.json using the
format defined in agent_docs/phase1_goals.md.
Print a summary of the results when done.
```

---

## Tips for Working with Claude Code on This Project

- **One week at a time.** Don't ask it to build multiple weeks at once.
- **Plan first, always.** Prompt ends with "wait for approval before coding."
- **Run pytest after each module.** If tests fail, fix before moving on.
- **Check Space-Track credentials first** (Week 3 onward):
  `echo $SPACETRACK_USER` — if blank, set them before starting.
- **If context gets long,** start a new Claude Code session and
  re-read CLAUDE.md + the relevant week's goals to reset context.
