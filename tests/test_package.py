import json
from lt3wmt26.package_ocelot import write_task_file


def _read(out):
    return [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]


def test_sc_field_order(tmp_path):
    rows = [{"dataset_id": "d", "id": "0001", "input_sentence": "s",
             "pred_incorrect": "CORRECT", "pred_corrected": "CORRECT"}]
    out = tmp_path / "x.jsonl"
    write_task_file(rows, "sc", str(out))
    keys = list(json.loads(out.read_text().splitlines()[0]).keys())
    assert keys == ["dataset_id", "id", "input_sentence", "pred_incorrect", "pred_corrected"]


def test_gc_field_order_matches_sc(tmp_path):
    rows = [{"dataset_id": "d", "id": "0002", "input_sentence": "s",
             "pred_incorrect": "x", "pred_corrected": "y"}]
    out = tmp_path / "gc.jsonl"
    write_task_file(rows, "gc", str(out))
    keys = list(json.loads(out.read_text().splitlines()[0]).keys())
    assert keys == ["dataset_id", "id", "input_sentence", "pred_incorrect", "pred_corrected"]


def test_mt_field_order(tmp_path):
    rows = [{"dataset_id": "d", "sent_id": "3", "source": "hi", "pred": "hallo"}]
    out = tmp_path / "mt.jsonl"
    write_task_file(rows, "mt", str(out))
    keys = list(json.loads(out.read_text().splitlines()[0]).keys())
    assert keys == ["dataset_id", "sent_id", "source", "pred"]


def test_qa_field_order_and_pred_cast_to_int(tmp_path):
    rows = [{"dataset_id": "d", "question_id": "7", "question": "q?", "pred": "2"}]
    out = tmp_path / "qa.jsonl"
    write_task_file(rows, "qa", str(out))
    row = json.loads(out.read_text().splitlines()[0])
    assert list(row.keys()) == ["dataset_id", "question_id", "question", "pred"]
    assert row["pred"] == 2 and isinstance(row["pred"], int)


def test_mr_field_order(tmp_path):
    rows = [{"dataset_id": "d", "id": "low-hsb-0", "question": "q?", "pred": "42"}]
    out = tmp_path / "mr.jsonl"
    write_task_file(rows, "mr", str(out))
    keys = list(json.loads(out.read_text().splitlines()[0]).keys())
    assert keys == ["dataset_id", "id", "question", "pred"]


def test_missing_prediction_defaults_to_correct_for_sc(tmp_path):
    rows = [{"dataset_id": "d", "id": "0003", "input_sentence": "s"}]  # no pred_* fields
    out = tmp_path / "sc2.jsonl"
    write_task_file(rows, "sc", str(out))
    row = json.loads(out.read_text().splitlines()[0])
    assert row["pred_incorrect"] == "CORRECT" and row["pred_corrected"] == "CORRECT"


def test_multiple_rows_preserve_order(tmp_path):
    rows = [{"dataset_id": "d", "id": "mr-hsb-0", "question": "a", "pred": "1"},
            {"dataset_id": "d", "id": "mr-hsb-1", "question": "b", "pred": "2"}]
    out = tmp_path / "multi.jsonl"
    write_task_file(rows, "mr", str(out))
    got = _read(out)
    assert [r["id"] for r in got] == ["mr-hsb-0", "mr-hsb-1"]
