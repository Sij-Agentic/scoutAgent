# Scout schemas quick reference

This folder stores machine-readable schemas used by Scout workflows.

- File: `scout_plan_v1.json`
  - Schema id: `scout_plan_v1`
  - Purpose: validate the JSON object emitted by `ScoutAgent.plan()`.

Key fields in plan object:
- `schema`: must equal `scout_plan_v1`.
- `version`: optional string (e.g., `"1.1"`) for minor schema extensions.
- `target_market`: human-readable string description.
- `time_window`: one of `3m|6m|12m|24m|all`.
- `sources`: array of strings (e.g., `reddit`).
- `keywords`: array of strings used to drive collection.
- `subreddits`: array of subreddit names when using Reddit.
- `limits`:
  - `per_query_limit`: integer, max posts per keyword.
  - `comment_depth`: integer depth of comment tree to include.
  - `comment_limit`: integer max comments per post.
  - `min_num_comments`: integer quality filter for posts.
  - `min_score`: integer quality filter for posts.
- `dag`:
  - `run_id`: identifier for this run; DAG artifacts are stored under `data/runs/{run_id}/`.
  - `nodes`: ordered list of node specs.

Node spec fields:
- `id`: unique node id inside the DAG.
- `type`: `agent` or `tool`.
- `agent`: optional agent name when `type=agent` (e.g., `scout_agent`).
- `stage`: optional stage for agents: `plan|think|act`.
- `tool`: optional exact MCP tool name when `type=tool` (e.g., `reddit_search_and_fetch_threads`).
- `params`: optional object of arguments for the tool call (validated by runtime dispatcher).
- `code`: optional string; a human-readable code snippet describing the call. Not executed.
- `parallelize_by`: optional key for fanning out (e.g., `keyword`).
- `inputs`: object of inputs for the node; used by DAG driver to call agent/tool.
- `outputs`: array of artifact names produced by the node (e.g., `reddit_index.json`).
- `deps`: array of node ids this node depends on.

Typical minimal DAG:
- plan (agent: scout, stage: plan)
- collect (tool: exact MCP tool name, depends on plan)
- think (agent: scout, stage: think, depends on collect)
- act (agent: scout, stage: act, depends on think)

Artifacts and locations:
- `data/runs/{run_id}/plan.json` — persisted plan output.
- `data/runs/{run_id}/reddit_index.json` — output from collect node (list of threads/posts + paths).
- `data/runs/{run_id}/scout_think.json` — analysis JSON from think.
- `data/runs/{run_id}/scout_output.json` — final act output.

MCP Reddit tools:
- `reddit_search_and_fetch_threads` — cache-backed (reads `data/reddit_cache/threads/*.json`).
- `reddit_api_search_and_fetch_threads` — API-backed via `sources/reddit_client.py` (uses cache for persistence).

v1.1 enhancements:
- Plan nodes can carry exact MCP `tool` names and optional `params` for deterministic dispatch.
- Optional `code` field can be included for transparency but is never required for execution.
- During planning, available tools are discovered via MCP and listed in the prompt to ensure the LLM only uses valid tool names.

Notes:
- Keep the plan small and deterministic; favor references to cached files for large data.
- Use `parallelize_by` to allow the DAG driver to fan-out and cap concurrency.
- All nodes should be idempotent and check for existing artifacts to enable checkpoint replay.
