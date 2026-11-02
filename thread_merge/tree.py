"""Minimal comment-tree format and tree restructuring.

This module has no LLM, embedding or network dependencies. It defines:

* the minimal JSON format ("mintree/1") that all inputs are converted to,
* the Reddit -> mintree converter,
* flattening of trees into per-node records,
* merge decisions and the union-find used to group merged nodes,
* building one merged tree from two threads plus a list of merge decisions,
* level-by-level sibling deduplication driven by an external decider callback,
* moving replies that no longer fit a merged node under a copy of their original parent.

Minimal JSON format ("mintree/1")
---------------------------------
::

    {
      "format": "mintree/1",
      "title": "UBI: yes or no?",
      "source": "https://www.reddit.com/r/AskReddit/comments/abc123/ubi_yes_or_no/",
      "root": {
        "id": "t3_abc123",
        "text": "UBI: yes or no?\\n\\n<post body>",
        "score": 812,
        "children": [
          {"id": "t1_def456", "text": "...", "score": 40, "children": []}
        ]
      }
    }

Node fields:

* ``id`` (str, required): unique within the file. Reddit fullnames ("t1_xxx") are suitable.
* ``text`` (str, required): comment body as written (Markdown).
* ``score`` (int, optional, default 0).
* ``children`` (list of nodes, optional, default []).
* ``merged_from`` (list of str, optional): only present in merged output; ids of every
  original node that was combined into this node.
* ``junk`` (bool, optional, default false): only in merged output; the comment was judged
  short, off-topic or unproductive, so it was never embedded or merged.
* ``replies_moved_to`` (str, optional): only in merged output. Marks a copy of a comment that
  was merged into the node with this id, but whose text was discarded there. The copy holds
  the replies that did not make sense under the merged node's text; the comment's other
  replies were moved to that node. The copy keeps the original comment's id (which also
  appears in the merged node's ``merged_from``).

The root node is the post itself. When a thread is loaded, every id receives a tree
prefix (e.g. "A/", "B/") so ids stay unique across both inputs; merged output keeps the
prefixed ids, so every merged node can be traced back to its source thread.

Top-level ``source`` (str, optional) is the permalink of the post the file was made from.
Together with Reddit fullname ids it locates every node on Reddit (``node_url``), so further
metadata (author, timestamps, edits) can be fetched later. Merged output has several
sources, so it writes ``sources`` instead: a map from id prefix to permalink, e.g.
``{"A/": "https://...", "B/": "https://..."}``.
"""

from __future__ import annotations

import html
import json
import logging
from collections.abc import Callable, Container, Hashable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

log = logging.getLogger(__name__)

FORMAT_TAG = "mintree/1"
CONCAT_SEPARATOR = "\n\n---\n\n"
REDDIT_BASE_URL = "https://www.reddit.com"

TextPolicy = Literal["best", "concat"]
"""How the text of a merged node is chosen.

* ``best``: the text of one member (a subsuming member if any, otherwise the highest score).
* ``concat``: that member's text followed by the other members' distinct texts.
"""

InputFormat = Literal["minimal", "reddit"]


# --------------------------------------------------------------------------- data model


@dataclass
class Node:
    id: str
    text: str
    score: int = 0
    children: list[Node] = field(default_factory=list)
    merged_from: list[str] = field(default_factory=list)
    junk: bool = False
    replies_moved_to: str | None = None

    def source_ids(self) -> list[str]:
        """Ids of the original nodes this node represents (itself if not merged)."""
        return list(self.merged_from) if self.merged_from else [self.id]


@dataclass
class Thread:
    title: str
    root: Node
    sources: dict[str, str] = field(default_factory=dict)
    """Id prefix -> source post permalink ("" for an unprefixed thread)."""


# --------------------------------------------------------------------------- (de)serialisation


def node_from_dict(data: Mapping[str, Any], prefix: str = "") -> Node:
    return Node(
        id=prefix + str(data["id"]),
        text=str(data.get("text", "")),
        score=int(data.get("score", 0)),
        children=[node_from_dict(child, prefix) for child in data.get("children", [])],
        merged_from=[prefix + str(m) for m in data.get("merged_from", [])],
        junk=bool(data.get("junk", False)),
        replies_moved_to=prefix + str(data["replies_moved_to"]) if data.get("replies_moved_to") else None,
    )


def node_to_dict(node: Node) -> dict[str, Any]:
    out: dict[str, Any] = {"id": node.id, "text": node.text, "score": node.score}
    if node.merged_from:
        out["merged_from"] = list(node.merged_from)
    if node.junk:
        out["junk"] = True
    if node.replies_moved_to:
        out["replies_moved_to"] = node.replies_moved_to
    out["children"] = [node_to_dict(child) for child in node.children]
    return out


def thread_from_dict(data: Mapping[str, Any], prefix: str = "") -> Thread:
    if data.get("format") != FORMAT_TAG:
        raise ValueError(f"expected format {FORMAT_TAG!r}, got {data.get('format')!r}")
    sources = {prefix + str(k): str(v) for k, v in data.get("sources", {}).items()}
    if data.get("source"):
        sources[prefix] = str(data["source"])
    thread = Thread(title=str(data.get("title", "")), root=node_from_dict(data["root"], prefix), sources=sources)
    _check_unique_ids(thread.root)
    return thread


def thread_to_dict(thread: Thread) -> dict[str, Any]:
    out: dict[str, Any] = {"format": FORMAT_TAG, "title": thread.title}
    if list(thread.sources) == [""]:
        out["source"] = thread.sources[""]
    elif thread.sources:
        out["sources"] = dict(thread.sources)
    out["root"] = node_to_dict(thread.root)
    return out


def node_url(thread: Thread, node_id: str) -> str | None:
    """Reddit permalink of a node (post or comment), or None if its source is unknown.

    Uses the source registered for the node's id prefix (the longest matching one) and the
    Reddit fullname after the prefix: "t3_..." is the post itself, "t1_<id>" is the comment
    at ``<post permalink><id>/``.
    """
    prefixes = [p for p in thread.sources if node_id.startswith(p)]
    if not prefixes:
        return None
    prefix = max(prefixes, key=len)
    base, fullname = thread.sources[prefix], node_id[len(prefix):]
    if fullname.startswith("t3_"):
        return base
    if fullname.startswith("t1_"):
        return base.rstrip("/") + "/" + fullname[3:] + "/"
    return None


def load_thread(path: str | Path, prefix: str = "", input_format: InputFormat = "minimal") -> Thread:
    """Load a thread file; ``input_format="reddit"`` converts a Reddit dump first."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if input_format == "reddit":
        data = reddit_to_minimal(data)
    return thread_from_dict(data, prefix)


def save_thread(thread: Thread, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(thread_to_dict(thread), fh, ensure_ascii=False, indent=1)


def reddit_to_minimal(reddit_json: Any) -> dict[str, Any]:
    """Convert a Reddit thread JSON dump into a mintree/1 dict.

    Expected input: the body of ``https://www.reddit.com/r/<sub>/comments/<post_id>.json``,
    i.e. a JSON array of two "Listing" objects::

        [
          {"kind": "Listing", "data": {"children": [{"kind": "t3", "data": {...post...}}]}},
          {"kind": "Listing", "data": {"children": [
              {"kind": "t1", "data": {...comment...}},
              {"kind": "more", "data": {"count": 12, "children": ["id1", ...]}}
          ]}}
        ]

    Implementation outline:

    1. Validate the shape (list of length 2, both ``kind == "Listing"``); raise ``ValueError``
       with a clear message otherwise.
    2. Post: ``post = reddit_json[0]["data"]["children"][0]["data"]``.
       Root node: ``id = post["name"]`` (fullname "t3_..."),
       ``text = post["title"]`` plus ``"\\n\\n" + post["selftext"]`` when selftext is non-empty,
       ``score = int(post.get("score", 0))``.
    3. Comments: iterate ``reddit_json[1]["data"]["children"]``. For each child with
       ``kind == "t1"``: ``id = data["name"]``, ``text = data["body"]``,
       ``score = int(data.get("score", 0))``. Replies are in ``data["replies"]``, which is
       either the empty string ``""`` (no replies) or a Listing of the same shape; recurse.
    4. Skip ``kind == "more"`` entries (unloaded comments); log their total ``data["count"]``
       at INFO level so the user knows the dump is incomplete.
    5. Deleted or removed comments (body ``"[deleted]"`` or ``"[removed]"``): drop them if they
       have no surviving children; otherwise keep them with their body text so replies keep
       a parent.
    6. Apply ``html.unescape`` to bodies, titles and selftext (Reddit's JSON escapes
       ``&amp;``, ``&lt;``, ``&gt;``).
    7. Do not prefix ids; ``thread_from_dict``/``load_thread`` add the tree prefixes.
    8. Return ``{"format": FORMAT_TAG, "title": <unescaped post title>, "root": <root dict>}``
       using the node dict layout documented at the top of this module (keys ``id``,
       ``text``, ``score``, ``children``).

    Reddit dumps nest at most a few dozen levels, so plain recursion is adequate.
    """
    if not (
        isinstance(reddit_json, list)
        and len(reddit_json) == 2
        and all(isinstance(x, dict) and x.get("kind") == "Listing" for x in reddit_json)
    ):
        raise ValueError("expected a Reddit thread dump: a JSON array of two Listing objects")
    try:
        post = reddit_json[0]["data"]["children"][0]["data"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Reddit dump has no post in its first Listing") from exc

    title = html.unescape(str(post.get("title", "")))
    selftext = html.unescape(str(post.get("selftext") or ""))
    more_count = 0

    def convert(listing: Any) -> list[dict[str, Any]]:
        nonlocal more_count
        if not isinstance(listing, dict):  # "" when there are no replies
            return []
        out: list[dict[str, Any]] = []
        for child in listing.get("data", {}).get("children", []):
            kind, data = child.get("kind"), child.get("data", {})
            if kind == "more":
                more_count += int(data.get("count", 0))
                continue
            if kind != "t1":
                continue
            replies = convert(data.get("replies"))
            body = html.unescape(str(data.get("body", "")))
            if body.strip() in _DELETED_BODIES and not replies:
                continue
            out.append({"id": str(data["name"]), "text": body,
                        "score": int(data.get("score") or 0), "children": replies})
        return out

    children = convert(reddit_json[1])
    if more_count:
        log.info("Reddit dump is incomplete: %d comments behind 'load more' links were skipped", more_count)
    root = {
        "id": str(post["name"]),
        "text": title + ("\n\n" + selftext if selftext.strip() else ""),
        "score": int(post.get("score") or 0),
        "children": children,
    }
    out: dict[str, Any] = {"format": FORMAT_TAG, "title": title}
    if post.get("permalink"):
        out["source"] = REDDIT_BASE_URL + str(post["permalink"])
    out["root"] = root
    return out


_DELETED_BODIES = frozenset({"[deleted]", "[removed]"})


def iter_nodes(root: Node) -> Iterator[Node]:
    """Pre-order traversal without recursion."""
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


def count_nodes(root: Node) -> int:
    return sum(1 for _ in iter_nodes(root))


def _check_unique_ids(root: Node) -> None:
    seen: set[str] = set()
    for node in iter_nodes(root):
        if node.id in seen:
            raise ValueError(f"duplicate node id {node.id!r}")
        seen.add(node.id)


# --------------------------------------------------------------------------- flattening


@dataclass(frozen=True)
class FlatNode:
    """One node of a thread with its position, used by the matching stages."""

    id: str
    tree: str
    text: str
    score: int
    depth: int
    parent_id: str | None
    ancestor_ids: tuple[str, ...]  # root first, excluding the node itself


def flatten(thread: Thread, tree: str) -> list[FlatNode]:
    out: list[FlatNode] = []
    stack: list[tuple[Node, tuple[str, ...]]] = [(thread.root, ())]
    while stack:
        node, ancestors = stack.pop()
        out.append(
            FlatNode(
                id=node.id,
                tree=tree,
                text=node.text,
                score=node.score,
                depth=len(ancestors),
                parent_id=ancestors[-1] if ancestors else None,
                ancestor_ids=ancestors,
            )
        )
        for child in reversed(node.children):
            stack.append((child, ancestors + (node.id,)))
    return out


# --------------------------------------------------------------------------- merge decisions


class Verdict(str, Enum):
    SAME = "same"
    A_SUBSUMES_B = "a_subsumes_b"
    B_SUBSUMES_A = "b_subsumes_a"


@dataclass(frozen=True)
class MergeDecision:
    """An accepted decision that nodes ``a`` and ``b`` make the same point."""

    a: str
    b: str
    verdict: Verdict
    confidence: float
    reason: str = ""

    def preferred_id(self) -> str | None:
        """The node whose text should represent the pair, if the verdict names one."""
        if self.verdict is Verdict.A_SUBSUMES_B:
            return self.a
        if self.verdict is Verdict.B_SUBSUMES_A:
            return self.b
        return None


K = TypeVar("K", bound=Hashable)


class UnionFind(Generic[K]):
    """Disjoint sets; the earliest-inserted item of a set is its representative."""

    def __init__(self, items: Iterable[K]) -> None:
        self._parent: dict[K, K] = {}
        self._order: dict[K, int] = {}
        for item in items:
            if item not in self._parent:
                self._order[item] = len(self._parent)
                self._parent[item] = item

    def find(self, x: K) -> K:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: K, b: K) -> K:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self._order[rb] < self._order[ra]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        return ra

    def groups(self) -> dict[K, list[K]]:
        out: dict[K, list[K]] = {}
        for item in self._parent:
            out.setdefault(self.find(item), []).append(item)
        return out


def combine_nodes(
    nodes: Sequence[Node], decisions: Iterable[MergeDecision], policy: TextPolicy
) -> Node:
    """Combine nodes judged equivalent into one node.

    The representative ("head") is a member that some decision marks as subsuming, otherwise
    the highest-scoring member (first one on ties). The combined node keeps the head's id, so
    claims and embeddings computed for the head remain addressable. Score is the sum of member
    scores; children are the union of all members' children.
    """
    if not nodes:
        raise ValueError("combine_nodes needs at least one node")
    if len(nodes) == 1:
        return nodes[0]
    ids = {n.id for n in nodes}
    preferred = {p for d in decisions if (p := d.preferred_id()) is not None and p in ids}
    pool = [n for n in nodes if n.id in preferred] or list(nodes)
    head = max(pool, key=lambda n: n.score)

    if policy == "best":
        text = head.text
    else:
        texts = [head.text]
        for n in nodes:
            if n is not head and n.text.strip() and n.text not in texts:
                texts.append(n.text)
        text = CONCAT_SEPARATOR.join(texts)

    sources: list[str] = []
    for n in [head, *(n for n in nodes if n is not head)]:
        for sid in n.source_ids():
            if sid not in sources:
                sources.append(sid)

    children = sorted((c for n in nodes for c in n.children), key=lambda c: -c.score)
    return Node(
        id=head.id,
        text=text,
        score=sum(n.score for n in nodes),
        children=children,
        merged_from=sources,
    )


# --------------------------------------------------------------------------- global merge


def build_merged_tree(
    threads: Sequence[tuple[str, Thread]],
    decisions: Sequence[MergeDecision],
    policy: TextPolicy = "best",
    title: str | None = None,
) -> Thread:
    """Merge several threads into one tree according to ``decisions``.

    * All roots are merged into one root.
    * Each decision unions its two nodes into a group. A union is refused when it would put a
      node and one of its own ancestors from the same thread into one group.
    * Each group becomes one node (``combine_nodes``). Its parent is the group containing the
      parent of the group's shallowest member (ties: earlier thread, earlier position). Parent
      groups therefore always have a strictly smaller minimum depth, so no cycles can arise.
    * Children of every member end up under the merged node, because their own groups resolve
      their parent to this group.

    Input threads are not modified.
    """
    if not threads:
        raise ValueError("at least one thread is required")

    flat: dict[str, FlatNode] = {}
    originals: dict[str, Node] = {}
    for label, thread in threads:
        for fn in flatten(thread, label):
            if fn.id in flat:
                raise ValueError(f"id {fn.id!r} occurs in several threads; load them with distinct prefixes")
            flat[fn.id] = fn
        for node in iter_nodes(thread.root):
            originals[node.id] = node
    order = {nid: i for i, nid in enumerate(flat)}

    uf: UnionFind[str] = UnionFind(flat)
    members: dict[str, list[str]] = {nid: [nid] for nid in flat}

    def try_union(a: str, b: str) -> bool:
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            return True
        if _nests(members[ra], members[rb], flat):
            return False
        combined = members.pop(ra) + members.pop(rb)
        members[uf.union(ra, rb)] = combined
        return True

    root_ids = [thread.root.id for _, thread in threads]
    for rid in root_ids[1:]:
        try_union(root_ids[0], rid)

    applied: list[MergeDecision] = []
    for d in decisions:
        if d.a not in flat or d.b not in flat:
            log.warning("decision refers to unknown id(s): %s, %s", d.a, d.b)
        elif flat[d.a].depth == 0 or flat[d.b].depth == 0:
            log.debug("ignoring decision on a root node: %s, %s", d.a, d.b)
        elif try_union(d.a, d.b):
            applied.append(d)
        else:
            log.info("refused merge of %s and %s: would merge a node with its own ancestor", d.a, d.b)

    decisions_by_group: dict[str, list[MergeDecision]] = {}
    for d in applied:
        decisions_by_group.setdefault(uf.find(d.a), []).append(d)

    group_nodes: dict[str, Node] = {}
    for rep, ms in members.items():
        shells = [replace(originals[m], children=[], merged_from=list(originals[m].merged_from))
                  for m in sorted(ms, key=order.__getitem__)]
        group_nodes[rep] = combine_nodes(shells, decisions_by_group.get(rep, []), policy)

    root_rep = uf.find(root_ids[0])
    for rep, ms in members.items():
        if rep == root_rep:
            continue
        shallowest = min(ms, key=lambda m: (flat[m].depth, order[m]))
        parent_id = flat[shallowest].parent_id
        assert parent_id is not None, "non-root node without a parent"
        group_nodes[uf.find(parent_id)].children.append(group_nodes[rep])

    root = group_nodes[root_rep]
    for node in iter_nodes(root):
        node.children.sort(key=lambda c: (-c.score, order.get(c.id, 0)))
    sources = {p: url for _, thread in threads for p, url in thread.sources.items()}
    return Thread(title=title if title is not None else threads[0][1].title, root=root, sources=sources)


def _nests(group_a: Sequence[str], group_b: Sequence[str], flat: Mapping[str, FlatNode]) -> bool:
    """True if some node of one group is an ancestor of a node of the other in the same thread."""
    for x in group_a:
        fx = flat[x]
        for y in group_b:
            fy = flat[y]
            if fx.tree == fy.tree and (x in fy.ancestor_ids or y in fx.ancestor_ids):
                return True
    return False


# --------------------------------------------------------------------------- sibling dedup

Family = tuple[Node, Sequence[Node]]
LevelDecider = Callable[[Sequence[Family]], Sequence[MergeDecision]]
"""Callback receiving all (parent, children) families of one tree level that have at least two
children, and returning merge decisions between siblings (``a``/``b`` are child node ids).
Receiving a whole level at once allows the caller to bundle many families into few LLM calls."""


def dedup_siblings(root: Node, decide: LevelDecider, policy: TextPolicy = "best") -> list[MergeDecision]:
    """Merge equivalent siblings, top-down, one level at a time (modifies the tree in place).

    Merging at one level unions the children of the merged siblings; those children are then
    deduplicated when the next level is processed. Decisions linking nodes of different
    families are ignored. Returns the decisions that were applied.
    """
    applied: list[MergeDecision] = []
    level = [root]
    while level:
        families: list[Family] = [(p, list(p.children)) for p in level if len(p.children) >= 2]
        decisions = list(decide(families)) if families else []

        owner: dict[str, Node] = {c.id: p for p, kids in families for c in kids}
        by_parent: dict[int, list[MergeDecision]] = {}
        for d in decisions:
            pa, pb = owner.get(d.a), owner.get(d.b)
            if pa is None or pa is not pb or d.a == d.b:
                log.warning("ignoring sibling decision outside one family: %s, %s", d.a, d.b)
                continue
            by_parent.setdefault(id(pa), []).append(d)

        for parent, kids in families:
            decs = by_parent.get(id(parent))
            if decs:
                parent.children = _merge_siblings(kids, decs, policy)
                applied.extend(decs)
        level = [c for p in level for c in p.children]
    return applied


def _merge_siblings(kids: Sequence[Node], decisions: Sequence[MergeDecision], policy: TextPolicy) -> list[Node]:
    by_id = {k.id: k for k in kids}
    uf: UnionFind[str] = UnionFind(by_id)
    for d in decisions:
        uf.union(d.a, d.b)
    decs_by_rep: dict[str, list[MergeDecision]] = {}
    for d in decisions:
        decs_by_rep.setdefault(uf.find(d.a), []).append(d)
    out = [combine_nodes([by_id[i] for i in ids], decs_by_rep.get(rep, []), policy)
           for rep, ids in uf.groups().items()]
    out.sort(key=lambda n: -n.score)
    return out


# --------------------------------------------------------------------------- reply fit


@dataclass(frozen=True)
class ReplyFitCheck:
    """A reply that now sits under a merged node whose text came from another comment.

    ``reply_id`` was written as a reply to ``discarded_id``, which was merged into
    ``merged_id``; the merged node shows ``retained_text`` instead of ``discarded_text``.
    """

    merged_id: str
    retained_text: str
    discarded_id: str
    discarded_text: str
    reply_id: str
    reply_text: str


ReplyFitDecider = Callable[[Sequence[ReplyFitCheck]], set[str]]
"""Callback receiving all checks of a tree and returning the ``reply_id``s that do NOT make
sense as replies to their ``retained_text``. Receiving all checks at once allows bundling."""


def split_unfitting_replies(
    root: Node, originals: Mapping[str, FlatNode], decide: ReplyFitDecider, skip: Container[str] = ()
) -> tuple[list[ReplyFitCheck], list[ReplyFitCheck]]:
    """Move replies that no longer fit their merged parent under a copy of their own parent.

    Only meaningful when merged nodes show one member's text (text policy "best"). For every
    merged node below the root, each child none of whose members originally replied to the
    retained comment is checked (``skip``: ids exempt from checking, e.g. junk replies such
    as "This.", which fit any comment making the same point). Children that fail are moved
    under a copy of the discarded comment they originally replied to (its original id, text
    and score, ``replies_moved_to`` = the merged node's id), placed right after the merged
    node. ``originals`` maps every original id to its FlatNode. Modifies the tree in place;
    returns (all checks, failed checks).
    """
    parent_of: dict[int, Node] = {}
    for node in iter_nodes(root):
        for child in node.children:
            parent_of[id(child)] = node

    checks: list[ReplyFitCheck] = []
    for merged in iter_nodes(root):
        if merged is root or not merged.merged_from or merged.replies_moved_to:
            continue
        discarded = set(merged.merged_from) - {merged.id}
        for child in merged.children:
            if child.id in skip or child.replies_moved_to:
                continue
            original_parents = [originals[m].parent_id for m in child.source_ids() if m in originals]
            if merged.id in original_parents:
                continue
            d = next((p for p in original_parents if p in discarded), None)
            if d is None or originals[d].text == merged.text:
                continue
            checks.append(ReplyFitCheck(merged.id, merged.text, d, originals[d].text, child.id, child.text))

    failed_ids = decide(checks) if checks else set()
    failed = [c for c in checks if c.reply_id in failed_ids]
    by_node: dict[str, dict[str, list[str]]] = {}
    for c in failed:
        by_node.setdefault(c.merged_id, {}).setdefault(c.discarded_id, []).append(c.reply_id)

    for merged in [n for n in iter_nodes(root) if n.id in by_node]:
        parent = parent_of[id(merged)]
        position = parent.children.index(merged) + 1
        for d, reply_ids in by_node[merged.id].items():
            moved = [c for c in merged.children if c.id in reply_ids]
            merged.children = [c for c in merged.children if c.id not in reply_ids]
            source = originals[d]
            copy = Node(id=d, text=source.text, score=source.score, children=moved, replies_moved_to=merged.id)
            parent.children.insert(position, copy)
            position += 1
            log.info("moved %d replies of %s to a copy instead of merged node %s", len(moved), d, merged.id)
    return checks, failed
