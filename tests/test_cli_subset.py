from __future__ import annotations

import json
from pathlib import Path

import pytest

from debate_v_majority.cli import _build_arg_parser, _make_dataset_subset
from debate_v_majority.cli import main_impl as cli_main_impl


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_cli_has_no_duplicate_long_flags():
    parser = _build_arg_parser()
    flags = [
        opt
        for action in parser._actions
        for opt in action.option_strings
        if opt.startswith("--")
    ]
    assert len(flags) == len(set(flags))


def test_make_dataset_subset_with_explicit_ids(tmp_path: Path):
    path = tmp_path / "dataset.jsonl"
    rows = [{"id": i, "value": i} for i in range(5)]
    _write_jsonl(path, rows)
    items, meta = _make_dataset_subset(
        dataset="gpqa",
        test_path=path,
        n=2,
        seed=123,
        ids=[0, 3],
        range_str=None,
    )
    assert [it.orig_id for it in items] == [0, 3]
    assert meta["subset_size"] == 2


def test_make_dataset_subset_negative_id_maps_to_last_row(tmp_path: Path):
    path = tmp_path / "dataset.jsonl"
    rows = [{"id": i, "value": i} for i in range(4)]
    _write_jsonl(path, rows)
    items, _ = _make_dataset_subset(
        dataset="gpqa",
        test_path=path,
        n=1,
        seed=123,
        ids=[-1],
        range_str=None,
    )
    assert items[0].orig_id == -1
    assert items[0].raw_task["id"] == 3


def test_make_dataset_subset_out_of_range_id_raises(tmp_path: Path):
    path = tmp_path / "dataset.jsonl"
    rows = [{"id": i, "value": i} for i in range(2)]
    _write_jsonl(path, rows)
    with pytest.raises(IndexError):
        _make_dataset_subset(
            dataset="gpqa",
            test_path=path,
            n=1,
            seed=123,
            ids=[99],
            range_str=None,
        )


def test_default_dataset_test_path_prefers_first_existing_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    p1 = tmp_path / "repo" / "data" / "gpqa" / "test.jsonl"
    p2 = tmp_path / "pkg" / "data" / "gpqa" / "test.jsonl"
    p3 = tmp_path / "legacy" / "data" / "gpqa" / "test.jsonl"
    p2.parent.mkdir(parents=True, exist_ok=True)
    p2.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        cli_main_impl,
        "_dataset_test_path_candidates",
        lambda dataset, source_file=None: [p1, p2, p3],
    )

    selected = cli_main_impl._default_dataset_test_path("gpqa")
    assert selected == p2


def test_dataset_test_path_candidates_include_repo_package_and_legacy(tmp_path: Path):
    source_file = tmp_path / "repo" / "src" / "debate_v_majority" / "cli" / "main_impl.py"
    cands = cli_main_impl._dataset_test_path_candidates("aime25", source_file=source_file)

    assert cands[0] == tmp_path / "repo" / "data" / "aime25" / "test.jsonl"
    assert cands[1] == tmp_path / "repo" / "src" / "debate_v_majority" / "data" / "aime25" / "test.jsonl"
    assert cands[2] == tmp_path / "repo" / "src" / "debate_v_majority" / "cli" / "data" / "aime25" / "test.jsonl"
