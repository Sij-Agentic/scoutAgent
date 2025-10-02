# Cloud Run SSE Timeout Fix

## Problem
Jobs fail after exactly 4 minutes with `httpcore.ReadTimeout` error during Reddit API calls via SSE (Server-Sent Events).

## Root Cause
Cloud Run's **default 240-second (4 minute) request timeout** terminates long-running SSE connections before Reddit data collection completes.

## Solution

### Option 1: Quick Fix (Update Existing Services)
Run the fix script to update timeout settings without redeployment:

```bash
cd /home/ubuntu/ScoutAgent
./fix_cloudrun_timeout.sh
```

This updates both services to use 3600s (1 hour) timeout.

### Option 2: Full Redeployment
Rebuild and redeploy with the updated configuration:

```bash
cd /home/ubuntu/ScoutAgent

# Build new image
gcloud builds submit --tag gcr.io/$PROJECT_ID/scout-agent .

# Deploy with updated script (includes timeout fixes)
./deploy_cloudrun.sh
```

### Option 3: Manual Console Update
1. Go to Cloud Console → Cloud Run
2. Select `scout-agent-worker` service
3. Click "Edit & Deploy New Revision"
4. Expand "Container, Variables & Secrets, Connections, Security"
5. Set **Request timeout** to `3600` seconds
6. Click "Deploy"
7. Repeat for `scout-agent-api` service

## Changes Made

### 1. Application Code (`scout_agent/agents/scout.py`)
```python
# Before
multi_client = MultiMCPClient(server_configs)

# After
multi_client = MultiMCPClient(
    server_configs,
    max_retries=5,
    connection_timeout=120,
    sse_read_timeout=3600  # 1 hour for long Reddit API calls
)
```

### 2. API Service (`api_service.py`)
```python
# Before
async with httpx.AsyncClient(timeout=900.0) as client:  # 15 min

# After
async with httpx.AsyncClient(timeout=3600.0) as client:  # 1 hour
```

### 3. Deployment Script (`deploy_cloudrun.sh`)
```bash
# Worker Service
--timeout 3600 \
--request-timeout 3600 \

# API Service
--timeout 3600 \
```

## Verification

Check current timeout settings:
```bash
# Worker service
gcloud run services describe scout-agent-worker \
  --region us-central1 \
  --format="value(spec.template.spec.timeoutSeconds)"

# API service
gcloud run services describe scout-agent-api \
  --region us-central1 \
  --format="value(spec.template.spec.timeoutSeconds)"
```

Both should return `3600`.

## Testing

Submit a test job:
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

Monitor logs:
```bash
# Worker logs
gcloud run services logs read scout-agent-worker --region us-central1 --limit 100

# API logs
gcloud run services logs read scout-agent-api --region us-central1 --limit 100
```

## Why This Works

| Component | Before | After | Impact |
|-----------|--------|-------|--------|
| Cloud Run Worker Timeout | 900s (15 min) | 3600s (1 hour) | Prevents platform-level termination |
| Cloud Run API Timeout | 300s (5 min) | 3600s (1 hour) | Allows long background tasks |
| MCP Client SSE Timeout | 300s (5 min) | 3600s (1 hour) | Handles slow Reddit API responses |
| httpx Client Timeout | 900s (15 min) | 3600s (1 hour) | API→Worker communication |

The **4-minute failure** was Cloud Run's default 240s timeout (when not explicitly set), which overrode all application-level timeouts.

## Cost Implications

- **Longer timeout** = potential for higher costs if jobs hang
- **Mitigation**: Set `--max-instances 5` to limit concurrent long-running jobs
- **Monitoring**: Watch for jobs exceeding expected duration

## Alternative: Async Job Queue

For production, consider:
1. API returns immediately with `job_id`
2. Worker polls job queue (Cloud Tasks, Pub/Sub)
3. No long-running HTTP connections
4. Better scalability and cost control

This would eliminate timeout concerns entirely.
