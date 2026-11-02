from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from thread_merge.config import preset
from thread_merge.llm_client import QuotaExhausted
from thread_merge.pipeline import Components, DebugWriter, build_components, merge_threads
from thread_merge.tree import ReplyFitCheck, Thread, iter_nodes, load_thread, node_url, save_thread

from .fakes import FakeEmbedder, FakeExtractor, FakeMatcher, FakeReplyFitter, RecordingEmbedder

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_end_to_end_with_fakes() -> None:
    a = load_thread(EXAMPLES / "post_a.json", "A/")
    b = load_thread(EXAMPLES / "post_b.json", "B/")
    comps = Components(FakeExtractor(), FakeEmbedder(), FakeMatcher(accept_sim=0.75), FakeReplyFitter())
    result = merge_threads(a, b, preset("B"), comps)

    pairs = {(d.a, d.b) for d in result.global_decisions}
    assert ("A/t1_a1", "B/t1_b1r1") in pairs
    inflation = next(n for n in iter_nodes(result.thread.root) if n.id == "A/t1_a1")
    # The "funded by taxes" replies were deduplicated as siblings; the two "This." replies
    # are junk, so they stay separate.
    assert len(inflation.children) == 3
    seen = [sid for n in iter_nodes(result.thread.root) for sid in n.source_ids()]
    assert len(seen) == len(set(seen)) == 11


def test_option_a_is_scaffold_only() -> None:
    comps = build_components(preset("A"), topic="UBI")
    with pytest.raises(NotImplementedError):
        comps.extractor.extract([], {})


def test_sources_survive_merge(tmp_path: Path) -> None:
    urls = {"A/": "https://www.reddit.com/r/x/comments/aaa/ubi/",
            "B/": "https://www.reddit.com/r/y/comments/bbb/ubi/"}
    paths = {}
    for prefix, name in (("A/", "post_a.json"), ("B/", "post_b.json")):
        data = json.loads((EXAMPLES / name).read_text(encoding="utf-8"))
        data["source"] = urls[prefix]
        paths[prefix] = tmp_path / name
        paths[prefix].write_text(json.dumps(data), encoding="utf-8")

    a, b = load_thread(paths["A/"], "A/"), load_thread(paths["B/"], "B/")
    comps = Components(FakeExtractor(), FakeEmbedder(), FakeMatcher(accept_sim=0.75), FakeReplyFitter())
    result = merge_threads(a, b, preset("B"), comps)
    save_thread(result.thread, tmp_path / "merged.json")
    assert json.loads((tmp_path / "merged.json").read_text(encoding="utf-8"))["sources"] == urls

    merged = load_thread(tmp_path / "merged.json")
    ids = [sid for n in iter_nodes(merged.root) for sid in n.source_ids()]
    assert len(ids) == 11  # including every member listed in merged_from
    for sid in ids:
        assert node_url(merged, sid) is not None
    assert node_url(merged, "A/t3_aaa") == urls["A/"]
    assert node_url(merged, "B/t1_b1r1") == urls["B/"] + "t1_b1r1"[3:] + "/"


def load_examples() -> tuple[Thread, Thread]:
    return load_thread(EXAMPLES / "post_a.json", "A/"), load_thread(EXAMPLES / "post_b.json", "B/")


def only_inflation_merge() -> FakeMatcher:
    """Merges A's top-level inflation comment with B's nested one, and nothing else, so B's
    "funded by taxes" reply stays a separate child of the merged node."""
    return FakeMatcher(accept_sim=0.0, allow={frozenset(("A/t1_a1", "B/t1_b1r1"))})


def test_junk_is_not_embedded_or_merged() -> None:
    embedder = RecordingEmbedder()
    comps = Components(FakeExtractor(), embedder, FakeMatcher(accept_sim=0.0), FakeReplyFitter())
    result = merge_threads(*load_examples(), preset("B"), comps)

    assert result.junk_ids == {"A/t1_a1r2", "B/t1_b1r1r2"}  # the two "This." replies
    assert not any("This." in t for t in embedder.texts)
    nodes = {n.id: n for n in iter_nodes(result.thread.root)}
    for jid in result.junk_ids:
        # accept_sim=0 would merge anything with a vector; junk still stays on its own.
        assert nodes[jid].junk and not nodes[jid].merged_from
    for d in result.global_decisions + result.sibling_decisions:
        assert not {d.a, d.b} & result.junk_ids
    assert sum(n.junk for n in nodes.values()) == 2


def test_unfitting_replies_move_to_copy(tmp_path: Path) -> None:
    fitter = FakeReplyFitter(fail={"B/t1_b1r1r1"})
    comps = Components(FakeExtractor(), FakeEmbedder(), only_inflation_merge(), fitter)
    result = merge_threads(*load_examples(), preset("B"), comps)

    # a1 (score 50) is retained over b1r1; b1r1's replies are checked, except the junk "This.".
    assert [(c.merged_id, c.discarded_id, c.reply_id) for c in fitter.checks] == \
        [("A/t1_a1", "B/t1_b1r1", "B/t1_b1r1r1")]
    assert [c.reply_id for c in result.moved_replies] == ["B/t1_b1r1r1"]

    save_thread(result.thread, tmp_path / "merged.json")
    root = load_thread(tmp_path / "merged.json").root
    ids = [c.id for c in root.children]
    merged = root.children[ids.index("A/t1_a1")]
    copy = root.children[ids.index("A/t1_a1") + 1]  # placed right after the merged node
    assert copy.id == "B/t1_b1r1" and copy.replies_moved_to == "A/t1_a1"
    assert copy.text.startswith("UBI would cause inflation") and not copy.merged_from
    assert [c.id for c in copy.children] == ["B/t1_b1r1r1"]
    assert "B/t1_b1r1" in merged.merged_from  # the merge itself is kept and recorded
    assert {c.id for c in merged.children} == {"A/t1_a1r1", "A/t1_a1r2", "B/t1_b1r1r2"}


def test_reply_fit_skipped_with_concat() -> None:
    settings = preset("B")
    settings.text_policy = "concat"
    fitter = FakeReplyFitter(fail={"B/t1_b1r1r1"})
    comps = Components(FakeExtractor(), FakeEmbedder(), only_inflation_merge(), fitter)
    result = merge_threads(*load_examples(), settings, comps)
    assert fitter.checks == [] and result.moved_replies == []


class _QuotaFitter:
    """Reply fitter that fails like a used-up daily quota."""

    def __init__(self) -> None:
        self.log: list[dict[str, Any]] = []

    def __call__(self, checks: Sequence[ReplyFitCheck]) -> set[str]:
        raise QuotaExhausted("judge-model", "GenerateRequestsPerDay")


def test_debug_files_written_before_a_mid_run_stop(tmp_path: Path) -> None:
    comps = Components(FakeExtractor(), FakeEmbedder(), only_inflation_merge(), _QuotaFitter())
    with pytest.raises(QuotaExhausted):
        merge_threads(*load_examples(), preset("B"), comps, debug=DebugWriter(tmp_path))

    claims = json.loads((tmp_path / "claims.json").read_text(encoding="utf-8"))
    assert len(claims) == 11
    assert json.loads((tmp_path / "junk.json").read_text(encoding="utf-8")) == ["A/t1_a1r2", "B/t1_b1r1r2"]
    judged = [json.loads(line) for line in (tmp_path / "judgements.jsonl").read_text(encoding="utf-8").splitlines()]
    assert judged and len(judged) == len(comps.matcher.log)  # one line per judged pair, streamed
    merges = json.loads((tmp_path / "merges.json").read_text(encoding="utf-8"))
    assert {(m["pass"], m["a"], m["b"]) for m in merges} == {("global", "A/t1_a1", "B/t1_b1r1")}
    assert (tmp_path / "reply_fit.jsonl").read_text(encoding="utf-8") == ""
