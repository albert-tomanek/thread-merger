from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from praw.models import MoreComments

from fetch_reddit import ExpansionProgress, submission_to_minimal
from thread_merge.tree import thread_from_dict


def _comment(name: str, body: str, score: int = 1, replies: list[Any] | None = None) -> Any:
    return SimpleNamespace(fullname=name, body=body, score=score, replies=replies or [])


def test_submission_to_minimal() -> None:
    sub = SimpleNamespace(
        fullname="t3_p", title="UBI & you", selftext="Thoughts?", score=9,
        permalink="/r/AskReddit/comments/p/ubi_you/",
        comments=[
            _comment("t1_a", "Yes > no", 5, [
                _comment("t1_a1", "[deleted]", 0, [_comment("t1_a1a", "orphan reply")]),
                _comment("t1_a2", "[removed]"),
            ]),
            _comment("t1_b", "[deleted]"),
        ],
    )
    out = submission_to_minimal(sub)  # type: ignore[arg-type]
    root = out["root"]
    assert out["title"] == "UBI & you" and root["text"] == "UBI & you\n\nThoughts?"
    [a] = root["children"]  # t1_b is deleted with no replies
    [a1] = a["children"]  # deleted parent kept because it has a reply; t1_a2 dropped
    assert a1["children"][0]["id"] == "t1_a1a"
    assert out["source"] == "https://www.reddit.com/r/AskReddit/comments/p/ubi_you/"
    assert thread_from_dict(out, "A/").root.id == "A/t3_p"


def _stub(sid: str) -> MoreComments:
    return MoreComments(None, {"count": 1, "children": [sid], "id": sid, "parent_id": "t1_a"})  # type: ignore[arg-type]


def test_expansion_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    # Expanding s1 yields two comments (one with a reply) and reveals stub s2.
    fetched = {"s1": [_comment("t1_x", "x", replies=[_comment("t1_y", "y")]), _comment("t1_z", "z"), _stub("s2")],
               "s2": [_comment("t1_w", "w")]}
    fake = lambda more, *, update=True: fetched[more.id]  # noqa: E731
    monkeypatch.setattr(MoreComments, "comments", fake)

    s1 = _stub("s1")
    with ExpansionProgress([_comment("t1_a", "a"), s1], reported=5, limit=None, log_every_s=0) as progress:
        assert (progress.loaded, progress.pending) == (1, 1)
        s2 = s1.comments(update=False)[2]
        assert (progress.requests, progress.loaded, progress.pending) == (1, 4, 1)
        s2.comments(update=False)
        assert (progress.requests, progress.loaded, progress.pending) == (2, 5, 0)
    assert MoreComments.comments is fake  # wrapper removed on exit
