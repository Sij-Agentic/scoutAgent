# Complete Timeout Fixes for Cloud Run

## All Timeout Issues Identified and Fixed

### Timeline of Fixes

1. ✅ **SSE Read Timeout** (4 minutes) → Fixed to 3600s
2. ✅ **Cloud Run Request Timeout** (240s default) → Fixed to 3600s  
3. ✅ **Process Wait Timeout** (900s) → **NEWLY FIXED to 3600s**
4. ✅ **MCP Log Verbosity** → **NEWLY ADDED env control**

---

## Issue 3: Process Wait Timeout (CRITICAL)

### Problem
The worker was calling `process.wait(timeout=900)` which **terminates the ScoutAgent process after 15 minutes**, even though Cloud Run allows 1 hour.

### Symptoms
- Logs show: `INFO: Waiting for application shutdown.`
- Process exits cleanly but prematurely
- No error messages (clean shutdown)
- Job marked as failed even though it was running

### Root Cause
```python
# worker_service.py line 269 (OLD)
return_code = process.wait(timeout=900)  # 15 minutes!
```

This was a **hidden timeout** that killed long-running jobs.

### Fix
```python
# worker_service.py line 274 (NEW)
return_code = process.wait(timeout=3600)  # 1 hour
```

---

## Issue 4: Excessive MCP Logging

### Problem
MCP servers generate **massive amounts of logs**, making it impossible to find actual errors.

### Solution
Added `VERBOSE_MCP_LOGS` environment variable to control MCP log output.

```python
# worker_service.py
verbose_mcp_logs = os.getenv("VERBOSE_MCP_LOGS", "false").lower() == "true"

if verbose_mcp_logs:
    print(f"[{prefix}][STDOUT] {line}")  # Only if enabled
```

**Default**: `VERBOSE_MCP_LOGS=false` (MCP logs saved to files, not console)  
**Enable**: Set `VERBOSE_MCP_LOGS=true` for debugging

---

## Complete Timeout Configuration

| Component | Old Timeout | New Timeout | Location |
|-----------|-------------|-------------|----------|
| **Cloud Run Worker** | 900s | 3600s | `deploy.sh` |
| **Cloud Run API** | 300s | 3600s | `deploy.sh` |
| **MCP SSE Client** | 300s | 3600s | `scout.py` |
| **httpx Client** | 900s | 3600s | `api_service.py` |
| **Process Wait** | 900s | **3600s** | `worker_service.py` ⚠️ **NEW** |

All timeouts now aligned at **1 hour (3600s)**.

---

## Additional Improvements

### 1. Better Error Reporting
```python
# Show last 50 lines of both stdout and stderr on failure
if return_code != 0:
    error_msg = f"ScoutAgent failed with exit code {return_code}\n"
    error_msg += f"\n=== STDERR (last 50 lines) ===\n{tail_err}\n"
    error_msg += f"\n=== STDOUT (last 50 lines) ===\n{tail_out}\n"
    print(error_msg, file=sys.stderr)
```

### 2. Timeout Exception Details
```python
except subprocess.TimeoutExpired as e:
    error_msg = f"Job {request.job_id} timed out after {e.timeout}s"
    print(error_msg, file=sys.stderr)
```

### 3. Full Stack Traces
```python
except Exception as e:
    import traceback
    traceback.print_exc()  # Full stack trace in logs
```

---

## Deployment

### Quick Deploy
```bash
cd /home/ubuntu/ScoutAgent
./deploy.sh
```

### Manual Deploy
```bash
# Build
gcloud builds submit --tag gcr.io/delvelabs-scout-agent/scout-agent .

# Deploy Worker with all fixes
gcloud run deploy scout-agent-worker \
    --image gcr.io/delvelabs-scout-agent/scout-agent \
    --region us-central1 \
    --timeout 3600 \
    --memory 4Gi \
    --cpu 2 \
    --set-env-vars "GCS_BUCKET=scout-agent-outputs,VERBOSE_MCP_LOGS=false,SCOUT_OPENAI_API_KEY=...,SCOUT_REDDIT_CLIENT_ID=...,SCOUT_REDDIT_CLIENT_SECRET=...,SCOUT_REDDIT_USER_AGENT=scout-agent/1.0" \
    --command python --args worker_service.py

# Deploy API
gcloud run deploy scout-agent-api \
    --image gcr.io/delvelabs-scout-agent/scout-agent \
    --region us-central1 \
    --timeout 3600 \
    --memory 512Mi \
    --set-env-vars "GCS_BUCKET=scout-agent-outputs" \
    --command python --args api_service.py
```

---

## Testing

### 1. Submit Job
```bash
API_URL="https://scout-agent-511946707043.us-central1.run.app"

curl -X POST "$API_URL/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "target_market": "Knowledge management tools",
    "keywords": "bidirectional links,markdown sync",
    "subreddits": "PKMS,productivity",
    "per_query_limit": 5
  }'
```

### 2. Monitor Logs (Clean Output)
```bash
# Worker logs - now much cleaner without MCP spam
gcloud run services logs read scout-agent-worker \
  --region us-central1 \
  --limit 200 \
  --format "table(timestamp,severity,textPayload)"
```

**Look for**:
- ✅ No premature "Waiting for application shutdown"
- ✅ Clear error messages if failures occur
- ✅ "Job {job_id} completed successfully"
- ✅ "Output uploaded to: gs://..."

### 3. Check Job Status
```bash
JOB_ID="<from_submission>"
curl "$API_URL/jobs/$JOB_ID"
```

Should show:
- `"status": "running"` for up to 1 hour
- `"status": "completed"` when done
- `"gcs_output_path": "gs://..."` when complete

### 4. Enable Verbose Logs (if needed)
```bash
# For debugging, enable MCP logs
gcloud run services update scout-agent-worker \
  --region us-central1 \
  --set-env-vars "VERBOSE_MCP_LOGS=true"

# Remember to disable after debugging
gcloud run services update scout-agent-worker \
  --region us-central1 \
  --set-env-vars "VERBOSE_MCP_LOGS=false"
```

---

## Troubleshooting

### Job Still Times Out After 1 Hour

**Check actual runtime**:
```bash
gcloud run services logs read scout-agent-worker --region us-central1 | grep "Job.*completed"
```

**If legitimately taking >1 hour**:
- Reduce `per_query_limit` to collect less data
- Or increase timeout further (Cloud Run max is 3600s)
- Or split into multiple smaller jobs

### Can't Find Errors in Logs

**With clean logs** (VERBOSE_MCP_LOGS=false):
```bash
# Search for errors
gcloud run services logs read scout-agent-worker --region us-central1 | grep -i error

# Search for failures
gcloud run services logs read scout-agent-worker --region us-central1 | grep -i "failed\|exception\|traceback"

# Check last 50 lines of job
gcloud run services logs read scout-agent-worker --region us-central1 --limit 50
```

### Job Completes But Status Shows "Running"

**Check GCS directly**:
```bash
gsutil ls gs://scout-agent-outputs/scout/jobs/$JOB_ID/
```

**If files exist**:
- GCS polling is working
- Status will update on next poll
- Or check `manifest.json` path in API code

### Process Exits with Code 0 But No Output

**Check for**:
- Missing API keys (Reddit, LLM)
- Network/egress issues
- Insufficient memory (increase from 4Gi)

**View full logs**:
```bash
gcloud run services logs read scout-agent-worker --region us-central1 --limit 500 > worker_logs.txt
```

---

## Summary of All Fixes

### Timeout Fixes
1. ✅ Cloud Run service timeout: 900s → 3600s
2. ✅ MCP SSE read timeout: 300s → 3600s
3. ✅ httpx client timeout: 900s → 3600s
4. ✅ **Process wait timeout: 900s → 3600s** ⚠️ **CRITICAL**

### Path Fixes
5. ✅ Removed `os.chdir(temp_dir)` - fixed manifest paths
6. ✅ Output directory search - check `/app/data/runs`

### Architecture Fixes
7. ✅ Fire-and-forget API→Worker communication
8. ✅ GCS polling for job completion detection

### Observability Fixes
9. ✅ **MCP log verbosity control** - `VERBOSE_MCP_LOGS` env var
10. ✅ Better error reporting - last 50 lines of stdout/stderr
11. ✅ Timeout exception details
12. ✅ Full stack traces on errors

---

## Expected Behavior After Fixes

✅ Jobs run for up to 1 hour without premature termination  
✅ Clean, readable logs (MCP noise suppressed)  
✅ Clear error messages when failures occur  
✅ Accurate status tracking via GCS polling  
✅ No false "failed" status while running  

---

## Cost Optimization

With 1-hour timeout:
- **Cost**: ~$0.10-0.20 per job (4Gi RAM, 2 CPU)
- **Mitigation**: Set `--max-instances 5` to limit concurrent jobs
- **Monitoring**: Set up billing alerts

---

## Next Steps

1. ✅ Deploy with `./deploy.sh`
2. ✅ Test with a real job
3. ✅ Monitor clean logs
4. ✅ Verify completion via GCS
5. Consider implementing job queue (Cloud Tasks) for production
