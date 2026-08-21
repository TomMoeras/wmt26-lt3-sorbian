#!/usr/bin/env python3
"""OCELoT per-task schema emission (Instr17a, per organiser README Sec Output format +
dummy_submission).

Ported from `scripts/phase6/ocelot_repackage.py`. That script's `main()` merges a 14-file
prediction bundle onto the official blindset rows (per test set, mirroring row identity --
`dataset_id` plus the task's id field -- verbatim from the blindset) and validates full
coverage / unique ids / UTF-8 cleanliness; that join + validation is cluster I/O orchestration
and is not part of this module. What is kept is the pure, testable core: the OFFICIAL
per-task field order the organizer's scorer expects. `rows` here are already-assembled
per-item dicts (however the caller built them); this function only re-shapes each row into
that exact key order and writes one JSON object per line.

Field order per task (dataset_id, id.. hypothesis-echo fields.. prediction field(s)):
  MT     -- dataset_id, sent_id, source, pred
  QA     -- dataset_id, question_id, question, pred   (pred cast to int, per source)
  SC/GC  -- dataset_id, id, input_sentence, pred_incorrect, pred_corrected
  MR     -- dataset_id, id, question, pred

Prediction field is `pred` (NOT `hypothesis`); input-echo fields come from the official test
row. Missing prediction fields default the same way the source's blindset-join did (`CORRECT`
for SC/GC, `""` for MT/MR pred) -- a safe fallback for an unmatched join, not a silent
mask of a real bug in this module's own logic.
"""
import json

FIELD_ORDER = {
    "mt": ["dataset_id", "sent_id", "source", "pred"],
    "qa": ["dataset_id", "question_id", "question", "pred"],
    "sc": ["dataset_id", "id", "input_sentence", "pred_incorrect", "pred_corrected"],
    "gc": ["dataset_id", "id", "input_sentence", "pred_incorrect", "pred_corrected"],
    "mr": ["dataset_id", "id", "question", "pred"],
}

_DEFAULTS = {"pred": "", "pred_incorrect": "CORRECT", "pred_corrected": "CORRECT"}


def write_task_file(rows, task, out_path):
    """Write `rows` as one JSON object per line, in the official field order for `task`."""
    fields = FIELD_ORDER[task]
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in rows:
            out = {k: (r.get(k, _DEFAULTS[k]) if k in _DEFAULTS else r[k]) for k in fields}
            if task == "qa":
                out["pred"] = int(out["pred"])
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
