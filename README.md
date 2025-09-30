# ScoutAgent

An agentic system that discovers market pain points, validates them via research, identifies gaps, proposes solutions, and produces executive-grade reports — end-to-end.

## What this project is
- **Multi-agent research workflow** orchestrated by a DAG engine
- **Production-minded architecture**: robust logging, manifest-based persistence, retries/timeouts, isolated tool execution
- **LLM-augmented system** with multiple backends, strict parsing, and defensive data handling

## Core Components
- **DAG Orchestrator** (`scout_agent/orchestration/`)
  - Builds and executes agent workflows as a DAG
  - Handles plan → collect → think → act stages per agent
  - Manages dependencies, retries, timeouts, and manifest writes
- **Services Registry** (`scout_agent/`)
  - Centralized access to: logging, memory/manifest manager, message bus, LLM manager, MCP client
  - Ensures consistent configuration and testability
- **Memory & Replay** (`scout_agent/memory/manifest_manager.py`)
  - JSON manifest for every run (inputs, stage outputs, errors, timing)
  - Enables replay, inspection, and partial reruns without re-executing the full pipeline
- **Sandboxed Code Execution** (via tools service)
  - Tool nodes execute Python snippets with a prelude (e.g., `mcp_call`, `save_to_manifest`)
  - Isolates side-effects; captures stdout/JSON for reliable downstream parsing
- **MCP Tool Calling** (`scout_agent/mcp_integration/`)
  - Standard interface to external tools (Reddit, web search, content extract/triage, vendor research)
  - Retries, rate limits, and structured outputs
- **LLM Backends** (`scout_agent/llm/`)
  - Multiple providers (DeepSeek, Claude, etc.) behind a common manager
  - Centralized timeouts, retries, and model selection via env
  - Robust JSON extraction with multi-strategy parsing and fallbacks
- **Enhanced Development Utilities** (`debug_*.py`, tests)
  - Isolated tests for writer HTML generation, Docker parsing, manifest integrity
  - Docker-targeted logs with clear prefixes for environment-specific debugging

## Agents and Responsibilities
- **ScoutAgent** (`agents/scout.py`)
  - Plan metadata generation → programmatic DAG construction
  - Collect Reddit threads/comments via tools; normalize structures; JSON cleanup
  - Think stage summarizes and extracts candidate pain points
- **ScreenerAgent** (`agents/screener.py`)
  - Filters and prioritizes pain points; applies cost-limiting when needed
- **ValidatorAgent** (`agents/validator.py`)
  - Executes targeted research per pain point; consolidates evidence
  - Normalizes tool outputs and fixes double-encoded/markdown-wrapped JSON
- **GapFinderAgent** (`agents/gap_finder.py`)
  - Generates discovery queries; identifies gaps/opportunities
  - Uses intelligent fallbacks when LLM JSON is malformed or missing
- **BuilderAgent** (`agents/builder.py`)
  - Synthesizes solutions and strategies from validated data and gaps
  - Robust parsing for markdown/format variations
- **WriterAgent** (`agents/writer.py`)
  - Produces comprehensive report plans and final HTML
  - Multiple HTML extraction strategies and graceful fallbacks

## Autonomy: what it means here
- **Today**
  - Orchestrator chooses which agent/stage to run based on the DAG
  - Agents decide how to parse, clean, and structure data; when to use fallbacks
  - Cost-limiting knobs (top-k, depth, vendors, timeouts) influence breadth vs. speed
- **Beyond agent selection** (design direction)
  - Agents dynamically deciding: how deep to pursue a topic, when to broaden/narrow scope, when to stop due to diminishing returns
  - Adaptive iteration based on uncertainty signals and evidence quality
  - Self-tuning prompts/tools based on observed failures (e.g., malformed JSON, empty results)

## Data Flow and Reliability
- Manifest-first persistence with clear stage keys (e.g., `stages.scout_collect.reddit`)
- Direct message passing preferred; manifest used as reliable fallback
- Defensive JSON parsing throughout (code-block extraction, BOM stripping, regex reconstruction)
- Builder/Writer handle list-or-dict variations and markdown noise

## MCP Tools (examples)
- `reddit_api_search_and_fetch_threads`: API-backed search + threads/comments
- `identify_vendors`, `vendor_research_batch`: vendor analysis
- `extract_content`, `triage_content`: web content processing
- Extensible: add domain tools under `mcp_integration/server/`

## Running
- Local: `python -m scout_agent.main ...` with required env (LLMs, Reddit)
- Docker supported; ensure environment variables are set inside the container (e.g., `SCOUT_REDDIT_CLIENT_ID`, `SCOUT_REDDIT_CLIENT_SECRET`, `SCOUT_REDDIT_USER_AGENT`)

## Next Steps
- **More agentic inputs**: accept natural-language intent; system derives sources, queries, and depth automatically
- **Richer toolset**: Google Trends API, SERP providers, vector stores, product review scrapers
- **Smarter autonomy**: stopping criteria, uncertainty-aware branching, evidence thresholds
- **Writer improvements**: fully leverage upstream data; richer visualizations; zero-HTML-failure path
- **Observability**: standardized run dashboards; structured metrics for stage quality and cost

### Memory and Knowledge
- **Manifest schema evolution**: standardize stage records (input, context, output, errors, timings) with stable IDs for reliable cross-run references
- **Runs index**: build an on-disk index of past runs by tags (market, date, success, failures, agent, stage) for quick retrieval
- **Error catalog**: persist normalized error objects (type, message, traceback hash, offending node, recovery taken) with frequency stats to guide auto-mitigation
- **Retrieval API**: add query helpers to fetch “most similar past failure” or “best prior success” given current stage+inputs (string/embedding similarity)
- **Cross-run memory**: surface prior validated pain points, vendors, strategies into current runs as optional priors with provenance
- **Embeddings & summaries**: maintain short summaries + embeddings for large artifacts (LLM outputs, collected content) to enable fast lookup
- **Cache policy**: explicit TTLs and invalidation rules per tool (e.g., Reddit fetch vs. vendor research) with opt-in pinning for reproducibility
- **Retention & GC**: policies for pruning old runs while preserving representative samples and error exemplars
- **Time-travel replay**: CLI to replay any stage with exact inputs/context; diff outputs; promote fixed paths to defaults

## Key Environment Variables
- LLMs: `SCOUT_LLM_DEFAULT_MODEL`, `SCOUT_LLM_HTTP_TIMEOUT`, backend API keys
- Reddit: `SCOUT_REDDIT_CLIENT_ID`, `SCOUT_REDDIT_CLIENT_SECRET`, `SCOUT_REDDIT_USER_AGENT`

---