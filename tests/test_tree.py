from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from thread_merge.tree import (
    MergeDecision, Node, Thread, Verdict, build_merged_tree, combine_nodes, dedup_siblings,
    iter_nodes, load_thread, reddit_to_minimal, thread_from_dict, thread_to_dict,
)

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def load_pair() -> list[tuple[str, Thread]]:
    return [("A", load_thread(EXAMPLES / "post_a.json", "A/")),
            ("B", load_thread(EXAMPLES / "post_b.json", "B/"))]


def ids_under(node: Node) -> list[str]:
    return [c.id for c in node.children]


def test_roundtrip() -> None:
    _, a = load_pair()[0]
    again = thread_from_dict(thread_to_dict(a))
    assert thread_to_dict(again) == thread_to_dict(a)


def test_roots_merged_and_cross_depth_merge() -> None:
    threads = load_pair()
    d = MergeDecision("A/t1_a1", "B/t1_b1r1", Verdict.SAME, 0.9)
    merged = build_merged_tree(threads, [d])
    root = merged.root
    assert set(root.merged_from) == {"A/t3_aaa", "B/t3_bbb"}
    # The merged node sits under the root (its shallowest member a1 has depth 1) ...
    m = next(c for c in root.children if c.id == "A/t1_a1")
    assert set(m.merged_from) == {"A/t1_a1", "B/t1_b1r1"}
    assert m.score == 65
    # ... and carries the children of both members.
    assert set(ids_under(m)) == {"A/t1_a1r1", "A/t1_a1r2", "B/t1_b1r1r1", "B/t1_b1r1r2"}
    b1 = next(c for c in root.children if c.id == "B/t1_b1")
    assert b1.children == []
    # Every original node appears exactly once.
    seen = [sid for n in iter_nodes(root) for sid in n.source_ids()]
    assert len(seen) == len(set(seen)) == 11


def test_subsumption_picks_text() -> None:
    a = Node("x", "short", 10)
    b = Node("y", "long and fuller", 1)
    d = MergeDecision("x", "y", Verdict.B_SUBSUMES_A, 0.9)
    assert combine_nodes([a, b], [d], "best").text == "long and fuller"
    assert combine_nodes([a, b], [], "best").text == "short"
    assert combine_nodes([a, b], [], "concat").text.startswith("short\n\n---\n\nlong")


def test_ancestor_merge_refused() -> None:
    threads = load_pair()
    # b1r1 ~ a1 and b1r1r1 ~ a1 would put b1r1 and its own child into one group.
    ds = [MergeDecision("A/t1_a1", "B/t1_b1r1", Verdict.SAME, 0.9),
          MergeDecision("A/t1_a1", "B/t1_b1r1r1", Verdict.SAME, 0.8)]
    merged = build_merged_tree(threads, ds)
    m = next(n for n in iter_nodes(merged.root) if n.id == "A/t1_a1")
    assert "B/t1_b1r1r1" not in m.merged_from


def test_crossed_merges_do_not_cycle() -> None:
    threads = load_pair()
    # a1 (depth 1) ~ b1r1r1 (depth 3); a1r1 (depth 2) ~ b1 (depth 1): crossed depths.
    ds = [MergeDecision("A/t1_a1", "B/t1_b1r1r1", Verdict.SAME, 0.9),
          MergeDecision("A/t1_a1r1", "B/t1_b1", Verdict.SAME, 0.9)]
    merged = build_merged_tree(threads, ds)
    seen = [sid for n in iter_nodes(merged.root) for sid in n.source_ids()]
    assert len(seen) == len(set(seen)) == 11  # traversal terminates and covers everything


def test_dedup_siblings() -> None:
    threads = load_pair()
    merged = build_merged_tree(threads, [MergeDecision("A/t1_a1", "B/t1_b1r1", Verdict.SAME, 0.9)])

    def decide(families: object) -> list[MergeDecision]:
        return [MergeDecision("A/t1_a1r1", "B/t1_b1r1r1", Verdict.SAME, 0.9),
                MergeDecision("A/t1_a1r2", "B/t1_b1r1r2", Verdict.SAME, 0.9)]

    applied = dedup_siblings(merged.root, decide)  # type: ignore[arg-type]
    assert len(applied) == 2
    m = next(c for c in merged.root.children if c.id == "A/t1_a1")
    assert len(m.children) == 2


def _listing(*children: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "Listing", "data": {"children": list(children)}}


def _comment(name: str, body: str, score: int = 1, replies: Any = "") -> dict[str, Any]:
    return {"kind": "t1", "data": {"name": name, "body": body, "score": score, "replies": replies}}


def test_reddit_to_minimal() -> None:
    dump = [
        _listing({"kind": "t3", "data": {"name": "t3_p", "title": "UBI &amp; you",
                                          "selftext": "Thoughts?", "score": 9,
                                          "permalink": "/r/x/comments/p/ubi/"}}),
        _listing(
            _comment("t1_a", "Yes &gt; no", 5, _listing(
                _comment("t1_a1", "[deleted]", 0, _listing(_comment("t1_a1a", "orphan reply"))),
                _comment("t1_a2", "[removed]"),
            )),
            _comment("t1_b", "[deleted]"),
            {"kind": "more", "data": {"count": 7, "children": ["x"]}},
        ),
    ]
    out = reddit_to_minimal(dump)
    assert out["format"] == "mintree/1" and out["title"] == "UBI & you"
    assert out["source"] == "https://www.reddit.com/r/x/comments/p/ubi/"
    root = out["root"]
    assert root["id"] == "t3_p" and root["text"] == "UBI & you\n\nThoughts?" and root["score"] == 9
    [a] = root["children"]  # t1_b is deleted with no replies, the "more" stub is skipped
    assert a["text"] == "Yes > no"
    [a1] = a["children"]  # deleted parent kept because it has a reply; t1_a2 dropped
    assert a1["id"] == "t1_a1" and a1["children"][0]["id"] == "t1_a1a"
    assert thread_from_dict(out, "A/").root.children[0].id == "A/t1_a"


def test_reddit_to_minimal_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        reddit_to_minimal([])
