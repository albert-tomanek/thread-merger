# thread_merge

Thematic merging of two Reddit comment threads into one tree. Equivalent comments are merged
regardless of their depth, their children are joined, and duplicate siblings are merged
recursively.

* **Option B** (implemented): claim extraction + embeddings + LLM judge, all through the
  Google Gemini API free tier (OpenAI-compatible endpoint). Mistral is available as a paid
  alternative (`--provider mistral`).
* **Option A** (scaffolded): local Ollama model + sentence-transformers + similarity
  thresholds. Classes exist with full implementation notes and raise `NotImplementedError`.

## Setup (Option B)

1. Sign in to https://aistudio.google.com with a Google account and click "Get API key"
   (no card needed). Your project's free-tier limits are shown at
   https://aistudio.google.com/rate-limit.
2. To fetch threads from Reddit: create a "script" app at https://www.reddit.com/prefs/apps
   (any redirect uri, e.g. `http://localhost:8080`). The client id is the string under the
   app name; the secret is labelled "secret".
3. Create a virtual environment and install the dependencies. `virtualenv` is used instead
   of `python3 -m venv` because only it generates `activate.nu` for nushell:

   ```
   virtualenv .venv
   .venv/bin/pip install -r requirements.txt
   ```

   Activate it with `source .venv/bin/activate` (bash / zsh) or
   `overlay use .venv/bin/activate.nu` (nushell); `deactivate` leaves it. On nushell ≥ 0.106
   the generated `activate.nu` (virtualenv 20.31) prints a deprecation warning for `get -i`;
   `sed -i 's/get -i /get -o /g' .venv/bin/activate.nu` silences it.

4. Set the keys (bash / zsh):

   ```
   export GEMINI_API_KEY=...
   export REDDIT_CLIENT_ID=...
   export REDDIT_CLIENT_SECRET=...
   export REDDIT_USER_AGENT="thread_merge/0.1 by u/<your username>"   # optional
   ```

   In nushell use `$env.GEMINI_API_KEY = "..."` etc.

### Free-tier budget

Google no longer publishes the free quotas; they are shown per project in AI Studio. Reported
figures (late 2026) are about 500 requests/day and 10/minute per Flash-Lite model, and about
1,000/day for embeddings. Quotas are per model and reset at midnight Pacific time, so
extraction (`gemini-3.1-flash-lite`) and judging (`gemini-3.5-flash-lite`) use different
models to get a quota each. With the default bundle sizes (80 comments per extraction
request, 40 pairs per judge request), a pair of threads with about 5,000 comments in total
needs roughly 65 extraction, 75 embedding and 150–400 judge requests (the judge count depends
on how many candidate pairs the embeddings produce), so it fits in one day. At the default
9 requests/minute a run takes 30–60 minutes.

While running, each stage is announced (`[3/6] global pass: ...`) and every batch logs its
progress with an estimate of the time left (`judge: 12/40 batches (30%), ~4m10s left`).
Waits are logged with their cause: a per-minute rate limit or a server/connection problem is
retried automatically (`... rate limit reached (per-minute quota); waiting 38s before retry
1/6`). A used-up daily quota stops the run at once with exit code 2, saying when the quota
resets; rerun the same command then. Every response is cached in `.cache/`, so it resumes
where it stopped. The run ends with a summary of time taken, API requests made and time
spent waiting. If per-minute waits are frequent, lower `--rpm`. Free-tier prompts may be used by Google
to improve its models, except for users in the UK, EEA and Switzerland.

## Step-by-step usage

The commands below assume the venv from Setup; replace `.venv/bin/python` with `python` if
it is activated.

**1. Fetch both threads** (`fetch_reddit.py`, PRAW). This expands every "load more" stub, so
the whole comment section is downloaded; large threads take a few minutes.

```
.venv/bin/python fetch_reddit.py https://www.reddit.com/r/AskReddit/comments/xwbf4v/is_universal_basic_income_a_good_idea_why_or_why/ -o post_a.json
.venv/bin/python fetch_reddit.py <url of the second post> -o post_b.json
```

`--more-limit N` expands at most N stubs, for a quicker partial download.

*Alternative without a Reddit app:* save `<post url>.json?limit=500&depth=10` from a
browser and pass `--input-format reddit` in step 2. This is limited to about 500 comments;
the number of skipped comments is logged.

**2. Merge them** (claim extraction, embedding, global pass, tree build, sibling pass):

```
.venv/bin/python -m thread_merge post_a.json post_b.json -o merged.json --debug-dir debug/
```

To try the pipeline first on the bundled toy threads:

```
.venv/bin/python -m thread_merge examples/post_a.json examples/post_b.json -o merged.json --debug-dir debug/
```

**3. Inspect the result.** `merged.json` is a mintree/1 tree; nodes with `merged_from` combine
comments from both posts. In `debug/` (written while the run progresses, so a run that
stops early still leaves everything up to that point; the `.jsonl` files grow one line per
judged item as each batch finishes, so `tail -f debug/judgements.jsonl` follows the judge):

* `claims.json`: the extracted claim, stance, relation and kind for each comment.
* `judgements.jsonl`: every candidate pair the judge saw, its tests, verdict and reason.
* `merges.json`: the merges that were applied, per pass.
* `junk.json`: ids of the comments treated as junk.
* `reply_fit.jsonl`: every reply-fit check, whether the reply fits, and the reason.

**4. Tune and rerun** (see Tuning below). API responses are cached in `.cache/`, so a rerun
with changed thresholds only spends quota on requests that changed. Example, keeping all
distinct texts of merged comments:

```
.venv/bin/python -m thread_merge post_a.json post_b.json -o merged.json --debug-dir debug/ --text-policy concat
```

**Tests** (offline, no keys needed):

```
.venv/bin/python -m pytest -q
```

## Command-line flags

| flag | meaning |
|---|---|
| `--option A\|B` | preset (default B) |
| `--provider gemini\|mistral` | API provider for the `api` backends (default gemini) |
| `--extractor api\|local`, `--embedder api\|local`, `--matcher llm_judge\|threshold`, `--reply-fit api\|local` | per-stage override; stages can be mixed |
| `--input-format minimal\|reddit` | `reddit` converts a Reddit `.json` dump with `tree.reddit_to_minimal` |
| `--text-policy best\|concat` | merged text: one member's text, or all distinct texts joined |
| `--rpm N` | request throttle |
| `--cache-dir`, `--no-cache` | API responses are cached on disk (default `.cache/`); reruns cost no quota |
| `--debug-dir` | writes `claims.json`, `judgements.jsonl` (every judged pair with reasons), `merges.json`, `junk.json`, `reply_fit.jsonl` |

Tests (offline, no API key): `python -m pytest -q`

## Minimal format (`mintree/1`)

```json
{"format": "mintree/1", "title": "UBI: yes or no?",
 "source": "https://www.reddit.com/r/AskReddit/comments/abc/ubi_yes_or_no/",
 "root": {"id": "t3_abc", "text": "post title + body", "score": 812,
          "children": [{"id": "t1_def", "text": "...", "score": 40, "children": []}]}}
```

`score`, `children` and `source` are optional. Merged output keeps the `A/` / `B/` prefixes
that are added on load, and adds these node fields where they apply:

* `merged_from`: all original ids combined into this node.
* `junk: true`: the comment is short, off-topic or unproductive; it was never embedded or
  merged and stays where it was.
* `replies_moved_to: "<merged node id>"`: this node is a copy of a comment that was merged
  into that node with its text discarded. It holds the replies that did not make sense under
  the retained text; its other replies were moved to the merged node. It keeps the original
  comment's id, which is also listed in the merged node's `merged_from`.

`source` is the post's permalink; both Reddit converters fill it in. Since ids are Reddit
fullnames, it locates every node, so more metadata (author, timestamps, edits) can be
fetched later: `tree.node_url(thread, node_id)` returns the post URL for `t3_…` ids and
`<permalink><comment id>/` for `t1_…` ids. A merged file has two posts, so it writes
`"sources": {"A/": "<permalink of A>", "B/": "<permalink of B>"}` instead, and `node_url`
works on it for every id, including those in `merged_from`.

## Pipeline

1. **Load and flatten** both threads; ids are prefixed `A/` and `B/`.
2. **Claim extraction** (`extraction.py`): comments are sent in bundles of up to 80 (48k
   characters). Each comment comes with its parent's text, and the model returns stance,
   relation, kind, a context-resolved one-sentence claim and a junk flag. If any comments
   are missing from a reply, they are split in half and retried, falling back to the raw
   text.
3. **Junk filter** (`extraction.is_junk`): a comment is junk if the model flagged it (short
   filler, non-sequitur, no contribution), if its kind is agreement, joke or meta, or if it
   has at most `junk_max_words` (2) words. Junk is never embedded, so no later stage can
   merge it; it stays in place in the output, marked `junk: true`.
4. **Embedding** of `"[stance] claim"` for the remaining comments (`embeddings.py`,
   gemini-embedding-001).
5. **Global pass** (`matching.py`):
   * The candidates are each node's top-3 neighbours in the other thread with cosine ≥ 0.70.
   * The LLM judge assesses the candidates in bundles of 40 pairs. It answers three yes/no
     tests (same stance, same function, same core point), gives a verdict
     (`same` / `a_subsumes_b` / `b_subsumes_a` / `distinct`) and a confidence score.
   * Any failed test forces `distinct`.
   * A greedy one-to-one selection is then applied, so a node joins at most one pair.
6. **Tree build** (`tree.build_merged_tree`):
   * The two roots are merged.
   * Each merged group is attached under the parent of its shallowest member. This rule
     cannot create cycles.
   * Children of all members end up under the merged node.
   * A node is never merged with its own ancestor.
7. **Sibling pass** (`tree.dedup_siblings`): the tree is processed top-down, one level at a
   time. All families on a level are judged in shared bundles, and equivalent siblings are
   merged together with their children.
8. **Reply fit** (`tree.split_unfitting_replies`, `matching.LLMReplyFitter`; text policy
   `best` only, since `concat` keeps every text). A merged node shows one member's text, so
   replies written to a discarded member are checked in bundles: does the reply still make
   sense under the retained text? It fails if it refers to wording, numbers, examples or
   questions only the discarded comment had. Failing replies move under a copy of the
   discarded comment, placed right after the merged node and marked `replies_moved_to`.
   Junk replies are not checked, since they fit any comment that makes the same point.

## Files

| file | content |
|---|---|
| `tree.py` | format, (de)serialisation, `reddit_to_minimal`, flattening, union-find, merging; no LLM dependencies |
| `config.py` | `Settings`, presets A/B, all thresholds and batch sizes |
| `llm_client.py` | OpenAI-compatible client: JSON-validated chat with one repair retry, embeddings, throttle, logged retries, daily-quota detection, disk cache |
| `progress.py` | per-stage progress and time-left estimates |
| `prompts.py` | extraction, judge and reply-fit prompts (the rubrics live here) |
| `extraction.py` | `ApiClaimExtractor` (B), `LocalClaimExtractor` (A, stub) |
| `embeddings.py` | `ApiEmbedder` (B), `LocalEmbedder` (A, stub) |
| `matching.py` | candidate retrieval, `LLMJudgeMatcher` (B), `ThresholdMatcher` (A, stub), one-to-one selection, `LLMReplyFitter` (B), `LocalReplyFitter` (A, stub) |
| `pipeline.py` | orchestration and CLI |
| `fetch_reddit.py` (project root) | standalone PRAW fetcher: full Reddit thread → mintree/1 |

## Not yet implemented

Each of these has a docstring that is sufficient to implement it without other context:

* `extraction.LocalClaimExtractor`: can delegate to `ApiClaimExtractor` via Ollama's
  OpenAI-compatible endpoint.
* `embeddings.LocalEmbedder`
* `matching.ThresholdMatcher`
* `matching.LocalReplyFitter`: can delegate to `LLMReplyFitter` via Ollama.

## Tuning

The similarity thresholds in `config.py` are starting values. After a first run, label 30–50
pairs from `debug/judgements.jsonl` and adjust `candidate_min_sim` / `sibling_min_sim`. If
the judge's verdicts are rejected (`accepted: false`) for pairs that look right, adjust
`judge_min_confidence` instead.

## Switching provider

`--provider gemini` (default) or `--provider mistral` selects one of the presets in
`config.PROVIDERS`. Mistral needs `MISTRAL_API_KEY` and, for new accounts, pay-as-you-go
credits: its free "Experiment" API tier is no longer offered to new organisations (late 2026).

Any other OpenAI-compatible endpoint works too: add an `ApiSettings` preset (`base_url`,
`api_key_env`, model names, `requests_per_minute`) to `PROVIDERS`. Groq has no embedding
models, so it must be combined with `--embedder local`. Similarity thresholds depend on the
embedding model, so retune them after switching (see Tuning).
