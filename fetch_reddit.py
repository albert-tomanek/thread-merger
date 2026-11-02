"""Fetch a complete Reddit thread with PRAW and save it in the mintree/1 format.

Unlike the public ``<post url>.json`` endpoint, which returns at most ~500 comments and
leaves the rest as "load more" stubs, this expands every stub, so the whole comment section
is downloaded (large threads take a while: each stub costs one API request).

Setup:

1. Create a "script" app at https://www.reddit.com/prefs/apps (free, any redirect uri,
   e.g. http://localhost:8080).
2. ``pip install praw``
3. Set ``REDDIT_CLIENT_ID`` (the string under the app name) and ``REDDIT_CLIENT_SECRET``.
   ``REDDIT_USER_AGENT`` is optional; Reddit asks for something like
   ``"thread_merge/0.1 by u/<your username>"``.

Usage::

    python fetch_reddit.py https://www.reddit.com/r/AskReddit/comments/xwbf4v/... -o post_a.json

The output can be passed straight to ``python -m thread_merge`` (default input format).
Access is read-only and needs no Reddit password.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import praw
from praw.models import Comment, MoreComments, Submission

from thread_merge.tree import FORMAT_TAG, REDDIT_BASE_URL, count_nodes, thread_from_dict

log = logging.getLogger("fetch_reddit")

DELETED_BODIES = frozenset({"[deleted]", "[removed]"})


def make_reddit() -> praw.Reddit:
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise SystemExit("set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET (see the top of fetch_reddit.py)")
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=os.environ.get("REDDIT_USER_AGENT", "thread_merge/0.1 (comment fetcher)"),
    )
    reddit.read_only = True
    return reddit


def fetch_submission(reddit: praw.Reddit, url: str, more_limit: int | None = None) -> Submission:
    """Load a post and expand its "load more" / "continue this thread" stubs.

    ``more_limit=None`` expands all of them; a number expands at most that many.
    """
    submission = reddit.submission(url=url)
    submission.comment_sort = "top"
    log.info("fetching %s: %r (%s comments reported)", submission.id, submission.title, submission.num_comments)
    with ExpansionProgress(submission.comments.list(), submission.num_comments, more_limit):
        skipped = submission.comments.replace_more(limit=more_limit)
    if skipped:
        log.info("left %d 'load more' stubs unexpanded (--more-limit)", len(skipped))
    return submission


def count_items(items: Iterable[Comment | MoreComments]) -> tuple[int, int]:
    """(comments, stubs) in ``items`` and all their nested replies."""
    comments = stubs = 0
    stack = list(items)
    while stack:
        item = stack.pop()
        if isinstance(item, MoreComments):
            stubs += 1
        else:
            comments += 1
            stack.extend(item.replies)
    return comments, stubs


class ExpansionProgress:
    """Logs progress and an ETA while ``replace_more`` expands stubs.

    ``replace_more`` must run as one call (with a limit, it deletes the stubs it skips), so
    progress is tracked by temporarily wrapping ``MoreComments.comments``, which makes one
    API request per stub. Expanding a stub can reveal further stubs, so the remaining count
    (and the ETA) can grow while it runs.
    """

    def __init__(self, initial: Iterable[Comment | MoreComments], reported: int,
                 limit: int | None, log_every_s: float = 10.0) -> None:
        self.loaded, self.pending = count_items(initial)
        self.reported = reported
        self.limit = limit
        self.log_every_s = log_every_s
        self.requests = 0
        self._original = MoreComments.comments

    def __enter__(self) -> ExpansionProgress:
        log.info("expanding %d 'load more' stubs (%d comments loaded so far)", self.pending, self.loaded)
        self._start = self._last_log = time.monotonic()
        original, progress = self._original, self

        def comments(more: MoreComments, *, update: bool = True) -> list[Comment | MoreComments]:
            result = original(more, update=update)
            progress.record(result)
            return result

        MoreComments.comments = comments  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc: object) -> None:
        MoreComments.comments = self._original  # type: ignore[method-assign]
        if self.requests:
            log.info("expanded %d stubs in %s; %d comments loaded",
                     self.requests, _duration(time.monotonic() - self._start), self.loaded)

    def record(self, result: Sequence[Comment | MoreComments]) -> None:
        new_comments, new_stubs = count_items(result)
        self.requests += 1
        self.loaded += new_comments
        self.pending += new_stubs - 1
        now = time.monotonic()
        if now - self._last_log < self.log_every_s:
            return
        self._last_log = now
        todo = self.pending if self.limit is None else min(self.pending, self.limit - self.requests)
        eta = (now - self._start) / self.requests * todo
        log.info("expanded %d stubs, %d left; %d/%d comments loaded; ~%s remaining",
                 self.requests, todo, self.loaded, self.reported, _duration(eta))


def _duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def comment_to_node(comment: Comment) -> dict[str, Any] | None:
    """mintree node dict for a comment and its replies, or None if it should be dropped.

    Deleted or removed comments are dropped unless they still have replies, in which case
    they are kept with their placeholder text so the replies keep a parent.
    """
    children = convert_comments(comment.replies)
    body = comment.body or ""
    if body.strip() in DELETED_BODIES and not children:
        return None
    return {"id": comment.fullname, "text": body, "score": int(comment.score or 0), "children": children}


def convert_comments(comments: Iterable[Comment | MoreComments]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for c in comments:
        if isinstance(c, MoreComments):  # only present when --more-limit stopped expansion
            continue
        node = comment_to_node(c)
        if node is not None:
            out.append(node)
    return out


def submission_to_minimal(submission: Submission) -> dict[str, Any]:
    """mintree/1 dict for an already-expanded submission (ids are unprefixed fullnames)."""
    title = submission.title or ""
    selftext = submission.selftext or ""
    root = {
        "id": submission.fullname,
        "text": title + ("\n\n" + selftext if selftext.strip() else ""),
        "score": int(submission.score or 0),
        "children": convert_comments(submission.comments),
    }
    return {"format": FORMAT_TAG, "title": title, "source": REDDIT_BASE_URL + submission.permalink, "root": root}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fetch a full Reddit thread into mintree/1 JSON.")
    p.add_argument("url", help="post URL (a trailing .json or query string is fine)")
    p.add_argument("-o", "--output", type=Path, help="output file (default: <post id>.json)")
    p.add_argument("--more-limit", type=int, default=None,
                   help="expand at most N 'load more' stubs (default: all)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("prawcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    submission = fetch_submission(make_reddit(), args.url, args.more_limit)
    data = submission_to_minimal(submission)
    thread = thread_from_dict(data)  # validates the format and id uniqueness
    output = args.output or Path(f"{submission.id}.json")
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    log.info("wrote %s (%d comments)", output, count_nodes(thread.root) - 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
