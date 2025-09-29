# Scout schemas quick reference

This folder stores machine-readable schemas used by Scout workflows.

- File: `scout_plan_v2.json` (current)
  - Schema id: `scout_plan_v2`
  - Purpose: validate the JSON object emitted by `ScoutAgent.plan()` and track execution status.

- File: `scout_plan_v1.json` (legacy)
  - Schema id: `scout_plan_v1`
  - Purpose: validate the JSON object emitted by `ScoutAgent.plan()`.

## Scout Plan v2 Schema

Key fields in plan object:
- `schema`: must equal `scout_plan_v2`.
- `version`: string (e.g., `"2.0"`) for schema version.
- `run_id`: identifier for this run; DAG artifacts are stored under `data/runs/{run_id}/`.
- `run_metadata`: execution metadata for the entire run.
  - `started_at`: ISO timestamp when the run started.
  - `updated_at`: ISO timestamp when the run was last updated.
  - `status`: one of `initialized|running|completed|failed`.
  - `overall_progress`: float between 0 and 1 indicating progress.
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
  - `nodes`: ordered list of node specs.
- `stages`: dictionary of stage outputs and status.

## Legacy v1 Schema

Key fields in v1 plan object:
- `schema`: must equal `scout_plan_v1`.
- `version`: optional string (e.g., `"1.1"`) for minor schema extensions.
- `target_market`: human-readable string description.
- `time_window`: one of `3m|6m|12m|24m|all`.
- `sources`: array of strings (e.g., `reddit`).
- `keywords`: array of strings used to drive collection.
- `subreddits`: array of subreddit names when using Reddit.
- `limits`: collection limits configuration.
- `dag`:
  - `run_id`: identifier for this run.
  - `nodes`: ordered list of node specs.

## Node spec fields (v2):
- `id`: unique node id inside the DAG.
- `type`: `agent` or `tool`.
- `agent`: optional agent name when `type=agent` (e.g., `scout_agent`).
- `stage`: optional stage for agents: `plan|think|act`.
- `tool`: optional exact MCP tool name when `type=tool` (e.g., `reddit_search_and_fetch_threads`).
- `params`: optional object of arguments for the tool call (validated by runtime dispatcher).
- `code`: optional string; a human-readable code snippet describing the call. Not executed.
- `parallelize_by`: optional key for fanning out (e.g., `keyword`).
- `inputs`: object of inputs for the node; used by DAG driver to call agent/tool.
- `deps`: array of node ids this node depends on.
- `status`: execution status of the node.
  - `state`: one of `pending|running|completed|failed`.
  - `started_at`: ISO timestamp when execution started.
  - `updated_at`: ISO timestamp when execution was last updated.
  - `duration_seconds`: execution duration in seconds.
  - `attempts`: number of execution attempts.
  - `retries_remaining`: number of retries remaining.
  - `error`: error details if execution failed.
- `metrics`: execution metrics for the node.
  - `tokens_used`: number of tokens used by LLM calls.
  - `cost`: estimated cost of execution.
  - `backend`: LLM backend used (e.g., `openai`, `claude`, `gemini`).
  - `model`: specific model used (e.g., `gpt-4`, `claude-3-opus`).
- `outputs`: output details for the node.
  - `location`: location of outputs in the manifest (e.g., `stages.node_id.data`).
  - `artifacts`: list of file artifacts produced by the node.
    - `path`: relative path to the artifact.
    - `type`: file type (e.g., `json`, `txt`).
    - `size_bytes`: size of the artifact in bytes.

## Typical minimal DAG:
- plan (agent: scout, stage: plan)
- collect (tool: exact MCP tool name, depends on plan)
- think (agent: scout, stage: think, depends on collect)
- act (agent: scout, stage: act, depends on think)

## Manifest Structure (v2):
- Single consolidated manifest at `data/runs/{run_id}/run_manifest.json`
- All stage outputs stored in `stages.{stage_id}.data` section
- Node status and metrics tracked in the node itself and mirrored in the stages section
- File artifacts referenced in `outputs.artifacts` with paths relative to run directory

## Stages Section Structure:
- Each stage has its own entry in the `stages` object keyed by node ID
- Stage entries contain:
  - `data`: Output data from the node execution
    - For collect nodes: Contains execution details with thread/comment counts and results
    - For plan nodes: Contains a reference to the root-level plan data to avoid duplication
    - For think/act nodes: Contains the full output of those stages
  - `status`: Current execution state (`pending|running|completed|failed`)
  - `updated_at`: ISO timestamp of last update
  - `duration_seconds`: Execution time in seconds
  - `artifacts`: List of file artifacts produced by the node
  - `error`: Error details if execution failed

## Legacy Artifacts (v1):
- `data/runs/{run_id}/plan.json` — persisted plan output.
- `data/runs/{run_id}/reddit_index.json` — output from collect node (list of threads/posts + paths).
- `data/runs/{run_id}/scout_think.json` — analysis JSON from think.
- `data/runs/{run_id}/scout_output.json` — final act output.

MCP Reddit tools:
- `reddit_search_and_fetch_threads` — cache-backed (reads `data/reddit_cache/threads/*.json`).
- `reddit_api_search_and_fetch_threads` — API-backed via `sources/reddit_client.py` (uses cache for persistence).

## v2.0 enhancements:
- Comprehensive execution tracking with status, metrics, and error details
- Single consolidated manifest for all stages and outputs
- Standardized output locations using manifest sections
- Support for tracking LLM backend and model usage
- Detailed artifact tracking with file paths and sizes
- Progress tracking for overall workflow and individual nodes

## v1.1 enhancements:
- Plan nodes can carry exact MCP `tool` names and optional `params` for deterministic dispatch.
- Optional `code` field can be included for transparency but is never required for execution.
- During planning, available tools are discovered via MCP and listed in the prompt to ensure the LLM only uses valid tool names.

## Notes:
- Keep the plan small and deterministic; favor references to cached files for large data.
- Use `parallelize_by` to allow the DAG driver to fan-out and cap concurrency.
- All nodes should be idempotent and check for existing artifacts to enable checkpoint replay.
- Use the `ManifestManager` class to standardize manifest operations.
- Inputs for agent nodes should reference manifest sections, not raw file paths.
