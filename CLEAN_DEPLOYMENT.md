# Clean Deployment - Final Configuration

## Changes Made

### ✅ Use Trafilatura (Full WebContentExtractor)
**File**: `scout_agent/mcp_integration/server/gap_finder_tools.py` line 14
```python
from scout_agent.sources.web_content_extractor import WebContentExtractor  # Full version with trafilatura
```

### ✅ Removed use_cache Parameter
**Files Modified**:
1. `scout_agent/mcp_integration/server/gap_finder_tools.py`
   - `search_links()` - Removed `use_cache` parameter, hardcoded to `False`
   - `extract_content()` - Removed `use_cache` parameter, hardcoded to `False`

2. `scout_agent/agents/gap_finder.py`
   - Removed `use_cache` from all DAG node inputs (4 locations)

### ✅ Simplified - No Caching
All tools now use **live data only**:
- `search_links` → `use_cache=False`
- `extract_content` → `use_cache=False`

---

## Deploy

```bash
cd /home/ubuntu/ScoutAgent

# Use force deploy to bypass Cloud Run cache
./force_deploy.sh
```

**After deployment, SET API KEYS**:
```bash
gcloud run services update scout-agent-worker --region us-central1 \
  --update-env-vars "SCOUT_OPENAI_API_KEY=sk-...,SCOUT_ANTHROPIC_API_KEY=sk-ant-...,SCOUT_DEEPSEEK_API_KEY=sk-...,SCOUT_REDDIT_CLIENT_ID=...,SCOUT_REDDIT_CLIENT_SECRET=...,SCOUT_REDDIT_USER_AGENT=scout-agent/1.0"
```

---

## Verify

### 1. Check Logs
```bash
gcloud run services logs read scout-agent-worker --region us-central1 --limit 100
```

**Should see**:
- ✅ "Extracting content from: https://..." (using trafilatura)
- ✅ NO "use_cache" errors
- ✅ NO "unexpected keyword argument" errors
- ✅ LLM backends initialized (if API keys set)

### 2. Test Job
```bash
API_URL="https://scout-agent-511946707043.us-central1.run.app"

curl -X POST "$API_URL/jobs" -H "Content-Type: application/json" -d '{
  "target_market": "Knowledge management tools",
  "keywords": "bidirectional links",
  "subreddits": "PKMS",
  "per_query_limit": 2
}'
```

### 3. Monitor Progress
```bash
JOB_ID="<from_response>"
watch -n 10 "curl -s $API_URL/jobs/$JOB_ID | jq"
```

---

## What's Fixed

### All Issues Resolved
1. ✅ SSE timeout (4 min) → 3600s
2. ✅ Cloud Run timeout → 3600s  
3. ✅ Process wait timeout (15 min) → 3600s
4. ✅ Manifest path resolution
5. ✅ Fire-and-forget job processing
6. ✅ MCP log verbosity control
7. ✅ **WebContentExtractor using trafilatura (not simple version)**
8. ✅ **use_cache removed completely**
9. ✅ **No superfluous functions - clean codebase**

---

## Common Issues

### "No LLM backends were successfully initialized"
**Cause**: API keys not set on worker service  
**Fix**: Run the `gcloud run services update` command above with your actual API keys

### Still seeing use_cache errors
**Cause**: Old Cloud Run revision still serving traffic  
**Fix**: Use `./force_deploy.sh` which creates new revision with timestamp tag

### Job status stuck on "running"
**Cause**: Fire-and-forget architecture - status updates via GCS polling  
**Fix**: Check worker logs for actual progress, status will update when manifest appears in GCS

---

## Files Changed

| File | Change |
|------|--------|
| `gap_finder_tools.py` | Import full WebContentExtractor, remove use_cache |
| `gap_finder.py` | Remove use_cache from DAG nodes |
| `worker_service.py` | Extended timeouts, log control |
| `api_service.py` | Fire-and-forget, GCS polling |
| `force_deploy.sh` | Cache busting deployment |

---

## Summary

✅ **Clean codebase** - No superfluous functions  
✅ **Trafilatura only** - Full version with all features  
✅ **No caching** - Always use live data  
✅ **All timeouts fixed** - 1 hour across all layers  
✅ **Ready to deploy** - Run `./force_deploy.sh`

**Remember to set API keys after deployment!**
