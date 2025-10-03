# Cloud Run Deployment Fixes

## Issues Fixed

### 1. ✅ SSE Timeout (4-minute failure)
**Problem**: Jobs failed after exactly 4 minutes with `httpcore.ReadTimeout`

**Root Cause**: Cloud Run's default 240s timeout was too short for Reddit API calls

**Solution**:
- Increased Cloud Run timeout to 3600s (1 hour)
- Increased MCP client SSE timeout to 3600s
- Increased httpx client timeout to 3600s

**Files Changed**:
- `scout_agent/agents/scout.py` - MCP client timeout
- `api_service.py` - httpx timeout
- `deploy_cloudrun.sh` - Cloud Run service timeouts

---

### 2. ✅ Manifest Not Found Error
**Problem**: `Manifest not found: data/runs/scout_20251002_134741/run_manifest.json`

**Root Cause**: Worker service was changing to a temp directory (`os.chdir(temp_dir)`), causing path resolution mismatch:
- Scout agent creates manifest at `/app/data/runs/{run_id}/run_manifest.json`
- Validator agent looks for manifest at `data/runs/{run_id}/run_manifest.json` (relative path)
- When working directory is temp_dir, relative paths don't resolve correctly

**Solution**:
- Removed `os.chdir(temp_dir)` from worker service
- Keep working directory at `/app` for consistent path resolution
- Updated output directory search to check `/app/data/runs` first
- Added `/app/data/runs` directory creation in Dockerfile

**Files Changed**:
- `worker_service.py` - Removed directory change, fixed output search
- `Dockerfile` - Added `/app/data/runs` directory creation

---

## Deployment Steps

### Option A: Quick Timeout Fix Only (No Rebuild)
If you already deployed and just need to fix timeouts:

```bash
./update_timeout.sh
```

### Option B: Full Fix (Recommended)
Deploy with all fixes including manifest path resolution:

```bash
# Set your project ID
export PROJECT_ID="delvelabs-scout-agent"
export REGION="us-central1"

# Build and push new image
gcloud builds submit --tag gcr.io/$PROJECT_ID/scout-agent .

# Deploy services
./deploy_cloudrun.sh
```

Or manually:

```bash
# Build
gcloud builds submit --tag gcr.io/$PROJECT_ID/scout-agent .

# Deploy Worker
gcloud run deploy scout-agent-worker \
    --image gcr.io/$PROJECT_ID/scout-agent \
    --region $REGION \
    --timeout 3600 \
    --memory 4Gi \
    --cpu 2 \
    --port 8080 \
    --command python \
    --args worker_service.py \
    --set-env-vars "GCS_BUCKET=scout-agent-outputs,SCOUT_OPENAI_API_KEY=...,SCOUT_REDDIT_CLIENT_ID=...,SCOUT_REDDIT_CLIENT_SECRET=...,SCOUT_REDDIT_USER_AGENT=..."

# Deploy API
gcloud run deploy scout-agent-api \
    --image gcr.io/$PROJECT_ID/scout-agent \
    --region $REGION \
    --timeout 3600 \
    --memory 512Mi \
    --cpu 1 \
    --port 8080 \
    --command python \
    --args api_service.py \
    --set-env-vars "GCS_BUCKET=scout-agent-outputs,WORKER_SERVICE_URL=https://scout-agent-worker-xxx.run.app"
```

---

## Verification

### 1. Check Timeouts
```bash
# Worker
gcloud run services describe scout-agent-worker \
  --region us-central1 \
  --format="value(spec.template.spec.timeoutSeconds)"

# API
gcloud run services describe scout-agent-api \
  --region us-central1 \
  --format="value(spec.template.spec.timeoutSeconds)"
```

Both should return: **3600**

### 2. Test Job Submission
```bash
API_URL="https://scout-agent-511946707043.us-central1.run.app"

curl -X POST "$API_URL/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "target_market": "Knowledge management tools",
    "keywords": "bidirectional links,markdown sync",
    "subreddits": "PKMS,productivity",
    "per_query_limit": 2
  }'
```

### 3. Monitor Logs
```bash
# Worker logs (watch for manifest creation and path resolution)
gcloud run services logs read scout-agent-worker \
  --region us-central1 \
  --limit 200 \
  --format "table(timestamp,severity,textPayload)"

# Look for:
# - "Writing stage scout_collect output to manifest at: /app/data/runs/..."
# - "Found output directory at: /app/data/runs"
# - No "Manifest not found" errors
```

### 4. Check Job Status
```bash
JOB_ID="<job_id_from_submission>"
curl "$API_URL/jobs/$JOB_ID"
```

### 5. Verify GCS Upload
```bash
gsutil ls gs://scout-agent-outputs/scout/jobs/$JOB_ID/
```

---

## Technical Details

### Path Resolution Flow

**Before (Broken)**:
1. Worker: `os.chdir(temp_dir)` → working dir = `/tmp/xyz`
2. Scout: Creates manifest at `/app/data/runs/{run_id}/run_manifest.json`
3. Validator: Looks for `data/runs/{run_id}/run_manifest.json` (relative to `/tmp/xyz`)
4. Result: File not found ❌

**After (Fixed)**:
1. Worker: Stays in `/app` directory
2. Scout: Creates manifest at `/app/data/runs/{run_id}/run_manifest.json`
3. Validator: Looks for `data/runs/{run_id}/run_manifest.json` (relative to `/app`)
4. Result: File found ✅

### Timeout Configuration Layers

| Component | Timeout | Purpose |
|-----------|---------|---------|
| Cloud Run Worker | 3600s | Platform-level request timeout |
| Cloud Run API | 3600s | API service timeout |
| MCP SSE Client | 3600s | SSE read timeout for tool calls |
| httpx Client | 3600s | API→Worker HTTP timeout |

All layers must be aligned to prevent premature termination.

---

## Troubleshooting

### Still Getting Manifest Errors?
Check logs for actual path being used:
```bash
gcloud run services logs read scout-agent-worker --region us-central1 | grep "manifest"
```

### Timeout Still Occurring?
Verify all timeout settings:
```bash
# Check Cloud Run config
gcloud run services describe scout-agent-worker --region us-central1 --format=yaml | grep timeout

# Check deployed code has updated timeouts
gcloud run services logs read scout-agent-worker --region us-central1 | grep "sse_read_timeout"
```

### Output Not Uploading to GCS?
Check output directory detection:
```bash
gcloud run services logs read scout-agent-worker --region us-central1 | grep "output directory"
```

---

## Cost Implications

- **Longer timeout** = potential for higher costs if jobs hang
- **Mitigation**: 
  - Set `--max-instances 5` to limit concurrent jobs
  - Monitor job duration and set alerts
  - Consider implementing job timeout at application level

---

## Next Steps

1. ✅ Deploy fixes using Option B
2. ✅ Test with a real job
3. ✅ Monitor logs for successful completion
4. Consider implementing:
   - Job queue (Cloud Tasks/Pub/Sub) for better scalability
   - Progress tracking and cancellation
   - Automatic cleanup of old job data
   - Cost monitoring and alerts
