"""End-to-end pipeline and command-line entry point.

Stages:
 1. Load both threads (ids prefixed "A/" and "B/") and flatten them.
 2. Extract a normalised claim for every comment (bundled requests).
 3. Embed the claims.
 4. Global pass: cross-thread candidate pairs (top-k cosine), judged, then one-to-one selection.
 5. Build the merged tree (tree.build_merged_tree).
 6. Sibling pass: top-down, level by level, candidates among siblings, judged, merged
    (tree.dedup_siblings); all families of a level are judged in shared bundled requests.
 7. Reply fit (text policy "best" only): replies now under a merged node whose text came
    from another comment are checked; those that no longer fit move under a copy of their
    original parent (tree.split_unfitting_replies).
 8. Junk comments (short, off-topic or unproductive; see extraction.is_junk) are skipped by
    stages 3-7 and marked ``junk`` in the output.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import PROVIDERS, Settings, preset
from .embeddings import ApiEmbedder, Embedder, LocalEmbedder, embed_mapping
from .extraction import ApiClaimExtractor, Claim, ClaimExtractor, LocalClaimExtractor, is_junk
from .llm_client import LLMClient, LLMError, QuotaExhausted, next_pacific_midnight
from .progress import format_duration
from .matching import (
    CandidatePair, LLMJudgeMatcher, LLMReplyFitter, LocalReplyFitter, MatchContext, Matcher,
    ReplyFitter, ThresholdMatcher, cross_candidates, select_one_to_one, sibling_candidates,
)
from .tree import (
    Family, MergeDecision, ReplyFitCheck, Thread, build_merged_tree, count_nodes, dedup_siblings,
    flatten, iter_nodes, load_thread, save_thread, split_unfitting_replies,
)

log = logging.getLogger(__name__)


@dataclass
class Components:
    extractor: ClaimExtractor
    embedder: Embedder
    matcher: Matcher
    reply_fitter: ReplyFitter
    client: LLMClient | None = None  # shared API client, if any stage uses the API


@dataclass
class RunResult:
    thread: Thread
    claims: dict[str, Claim]
    global_decisions: list[MergeDecision]
    sibling_decisions: list[MergeDecision]
    judge_log: list[dict[str, Any]] = field(default_factory=list)
    junk_ids: set[str] = field(default_factory=set)
    moved_replies: list[ReplyFitCheck] = field(default_factory=list)
    reply_fit_log: list[dict[str, Any]] = field(default_factory=list)


def build_components(settings: Settings, topic: str) -> Components:
    client: LLMClient | None = None

    def api() -> LLMClient:
        nonlocal client
        if client is None:
            client = LLMClient.from_settings(settings.api, settings.cache_dir)
        return client

    extractor: ClaimExtractor
    if settings.extractor == "api":
        extractor = ApiClaimExtractor(
            api(), model=settings.api.extract_model, topic=topic,
            batch_size=settings.extract_batch_size, batch_chars=settings.extract_batch_chars,
            max_text_chars=settings.max_text_chars, max_parent_chars=settings.max_parent_chars,
        )
    else:
        extractor = LocalClaimExtractor(
            settings.local, topic=topic, cache_dir=settings.cache_dir,
            max_text_chars=settings.max_text_chars, max_parent_chars=settings.max_parent_chars,
        )

    embedder: Embedder
    if settings.embedder == "api":
        embedder = ApiEmbedder(api(), model=settings.api.embed_model, batch_size=settings.embed_batch_size)
    else:
        embedder = LocalEmbedder(settings.local.sentence_transformer_model)

    matcher: Matcher
    if settings.matcher == "llm_judge":
        matcher = LLMJudgeMatcher(
            api(), model=settings.api.judge_model, topic=topic,
            batch_size=settings.judge_batch_size, min_confidence=settings.judge_min_confidence,
            max_text_chars=settings.max_text_chars, max_parent_chars=settings.max_parent_chars,
        )
    else:
        matcher = ThresholdMatcher(settings.threshold_accept_sim)

    reply_fitter: ReplyFitter
    if settings.reply_fit == "api":
        reply_fitter = LLMReplyFitter(
            api(), model=settings.api.judge_model, topic=topic, batch_size=settings.judge_batch_size,
            max_text_chars=settings.max_text_chars, max_parent_chars=settings.max_parent_chars,
        )
    else:
        reply_fitter = LocalReplyFitter(settings.local, topic=topic, cache_dir=settings.cache_dir)
    return Components(extractor, embedder, matcher, reply_fitter, client)


def merge_threads(
    thread_a: Thread, thread_b: Thread, settings: Settings, components: Components | None = None,
    debug: DebugWriter | None = None,
) -> RunResult:
    """Merge two loaded threads. Ids must already be unique across both (use distinct prefixes).

    With ``debug``, each debug file is written as soon as its stage produces it, so a run that
    stops early (e.g. on a used-up daily quota) still leaves everything up to that point.
    """
    threads = [("A", thread_a), ("B", thread_b)]
    flat = [fn for label, th in threads for fn in flatten(th, label)]
    index = {fn.id: fn for fn in flat}
    topic = thread_a.title or thread_b.title
    comps = components or build_components(settings, topic)
    started = time.monotonic()
    if debug is not None:
        comps.matcher.log = debug.jsonl_log("judgements.jsonl")
        comps.reply_fitter.log = debug.jsonl_log("reply_fit.jsonl")

    comments = [fn for fn in flat if fn.depth > 0]
    log.info("%d comments in A, %d in B", sum(fn.tree == "A" for fn in comments),
             sum(fn.tree == "B" for fn in comments))

    stage(1, "claim extraction")
    claims = comps.extractor.extract(comments, index)
    for fn in comments:  # enforce the extractor contract
        if fn.id not in claims:
            claims[fn.id] = Claim(id=fn.id, claim=" ".join(fn.text.split())[:300] or "(empty)")
    for fn in flat:
        if fn.depth == 0:
            claims[fn.id] = Claim(id=fn.id, stance="neutral", relation="top_level", kind="other", claim=topic)
    if debug is not None:
        debug.write_json("claims.json", {k: v.model_dump() for k, v in claims.items()})

    stage(2, "junk filter and embedding")
    junk_ids = {fn.id for fn in comments if is_junk(claims[fn.id], fn.text, settings.junk_max_words)}
    log.info("%d of %d comments are junk and will not be embedded or merged", len(junk_ids), len(comments))
    if debug is not None:
        debug.write_json("junk.json", sorted(junk_ids))
    # Junk has no vector, so neither candidate retrieval (global or sibling) can select it.
    vectors = embed_mapping(comps.embedder, {fn.id: claims[fn.id].embedding_text()
                                             for fn in comments if fn.id not in junk_ids})
    ctx = MatchContext(
        claims=claims,
        texts={fn.id: fn.text for fn in flat},
        parent_texts={fn.id: (index[fn.parent_id].text if fn.parent_id and index[fn.parent_id].depth > 0 else None)
                      for fn in flat},
    )

    def eligible(tree: str) -> list[str]:
        return [fn.id for fn in comments if fn.tree == tree and fn.id not in junk_ids]

    pairs = cross_candidates(eligible("A"), eligible("B"), vectors,
                             top_k=settings.candidate_top_k, min_sim=settings.candidate_min_sim)
    stage(3, f"global pass: judging {len(pairs)} candidate pairs across the two threads")
    global_decisions = select_one_to_one(comps.matcher.decide(pairs, ctx))
    log.info("global pass: %d merges accepted", len(global_decisions))
    if debug is not None:
        debug.write_merges(global_decisions, [])

    stage(4, "building the merged tree")
    merged = build_merged_tree(threads, global_decisions, settings.text_policy)

    stage(5, "sibling pass (level by level, top-down)")
    level = 0

    def decide_level(families: Sequence[Family]) -> list[MergeDecision]:
        nonlocal level
        level += 1
        level_pairs: list[CandidatePair] = []
        for _parent, kids in families:
            level_pairs.extend(sibling_candidates([k.id for k in kids], vectors,
                                                  top_k=settings.candidate_top_k, min_sim=settings.sibling_min_sim))
        if not level_pairs:
            return []
        log.info("sibling pass, level %d: %d candidate pairs in %d families", level, len(level_pairs), len(families))
        return comps.matcher.decide(level_pairs, ctx)

    sibling_decisions = dedup_siblings(merged.root, decide_level, settings.text_policy)
    log.info("sibling pass: %d merges", len(sibling_decisions))
    if debug is not None:
        debug.write_merges(global_decisions, sibling_decisions)

    moved: list[ReplyFitCheck] = []
    stage(6, "reply fit" if settings.text_policy == "best" else "reply fit (skipped: text policy concat)")
    if settings.text_policy == "best":
        checks, moved = split_unfitting_replies(merged.root, index, comps.reply_fitter, skip=junk_ids)
        log.info("reply fit: %d replies checked, %d moved to copies of their original parent",
                 len(checks), len(moved))
    for node in iter_nodes(merged.root):
        node.junk = node.id in junk_ids
    log.info("merged tree has %d nodes (inputs: %d)", count_nodes(merged.root), len(flat))
    summary = f"finished in {format_duration(time.monotonic() - started)}"
    if comps.client is not None:
        summary += (f"; {comps.client.requests_made} API requests (others answered from cache), "
                    f"{format_duration(comps.client.seconds_waited)} spent waiting on rate limits or server errors")
    log.info(summary)
    return RunResult(merged, claims, global_decisions, sibling_decisions, list(comps.matcher.log),
                     junk_ids, moved, list(comps.reply_fitter.log))


STAGES = 6


def stage(n: int, name: str) -> None:
    log.info("[%d/%d] %s", n, STAGES, name)


class JsonlLog(list[dict[str, Any]]):
    """A list that also appends every entry to a JSON-lines file as it is added."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        path.write_text("", encoding="utf-8")

    def append(self, entry: dict[str, Any]) -> None:
        super().append(entry)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


class DebugWriter:
    """Writes the debug files while a run progresses (see ``merge_threads``).

    * ``claims.json`` after claim extraction; ``junk.json`` after the junk filter.
    * ``judgements.jsonl`` / ``reply_fit.jsonl``: one line per judged pair / checked reply,
      appended as each batch finishes (follow with ``tail -f``).
    * ``merges.json`` after the global pass, rewritten after the sibling pass.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        for stale in ("claims.json", "junk.json", "merges.json"):  # left over from an earlier run
            (directory / stale).unlink(missing_ok=True)

    def jsonl_log(self, name: str) -> JsonlLog:
        return JsonlLog(self.directory / name)

    def write_json(self, name: str, data: Any) -> None:
        path = self.directory / name
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)

    def write_merges(self, global_decisions: Sequence[MergeDecision], sibling_decisions: Sequence[MergeDecision]) -> None:
        self.write_json("merges.json", [
            {"pass": name, "a": d.a, "b": d.b, "verdict": d.verdict.value, "confidence": d.confidence, "reason": d.reason}
            for name, ds in (("global", global_decisions), ("sibling", sibling_decisions)) for d in ds
        ])


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="thread_merge", description="Merge two comment threads thematically.")
    p.add_argument("thread_a", type=Path)
    p.add_argument("thread_b", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True, help="merged mintree/1 JSON file")
    p.add_argument("--option", choices=["A", "B"], default="B", help="preset: A = local, B = API + LLM judge")
    p.add_argument("--provider", choices=list(PROVIDERS), default="gemini", help="API provider for the api backends")
    p.add_argument("--extractor", choices=["api", "local"])
    p.add_argument("--embedder", choices=["api", "local"])
    p.add_argument("--matcher", choices=["llm_judge", "threshold"])
    p.add_argument("--reply-fit", choices=["api", "local"])
    p.add_argument("--input-format", choices=["minimal", "reddit"], default="minimal")
    p.add_argument("--text-policy", choices=["best", "concat"])
    p.add_argument("--rpm", type=float, help="max API requests per minute")
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--debug-dir", type=Path, help="write claims, judgements and applied merges here")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    settings = preset(args.option)
    settings.api = PROVIDERS[args.provider]()
    if args.extractor:
        settings.extractor = args.extractor
    if args.embedder:
        settings.embedder = args.embedder
    if args.matcher:
        settings.matcher = args.matcher
    if args.reply_fit:
        settings.reply_fit = args.reply_fit
    if args.text_policy:
        settings.text_policy = args.text_policy
    if args.rpm is not None:
        settings.api.requests_per_minute = args.rpm
    if args.cache_dir:
        settings.cache_dir = args.cache_dir
    if args.no_cache:
        settings.cache_dir = None

    thread_a = load_thread(args.thread_a, prefix="A/", input_format=args.input_format)
    thread_b = load_thread(args.thread_b, prefix="B/", input_format=args.input_format)
    debug = DebugWriter(args.debug_dir) if args.debug_dir else None
    try:
        result = merge_threads(thread_a, thread_b, settings, debug=debug)
    except QuotaExhausted as exc:
        reset = next_pacific_midnight()
        wait = reset - reset.now(reset.tzinfo)
        log.error("stopped: %s", exc)
        log.error("The daily quota for %s is used up. Gemini quotas reset at midnight Pacific time "
                  "(%s local, in about %s). Rerun the same command then: finished requests are cached "
                  "in %s and will not be sent again.", exc.model, reset.astimezone().strftime("%H:%M"),
                  format_duration(wait.total_seconds()), settings.cache_dir or "(cache disabled!)")
        return 2
    except LLMError as exc:
        log.error("stopped: %s", exc)
        return 1
    save_thread(result.thread, args.output)
    log.info("wrote %s", args.output)
    return 0
