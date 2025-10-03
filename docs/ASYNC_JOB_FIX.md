# Async Job Processing Fix

## Problem
Jobs were being marked as "failed" prematurely even though the worker was still processing them.

### Symptoms
1. API returns `status: "failed"` with `error_message: None`
2. Worker logs show job still running successfully
3. Log streaming errors: `ValueError: I/O operation on closed file`

### Root Cause
The API service was using **synchronous HTTP wait** for worker completion:
- API sends request to worker
- API waits for HTTP response (up to 3600s)
- FastAPI BackgroundTasks has limitations for long operations
- Connection can timeout/fail even with high timeout settings
- Worker continues running but API thinks it failed

---

## Solution: Fire-and-Forget + GCS Polling

### Architecture Change

**Before (Synchronous)**:
```
API → Worker (wait 3600s for response) → Update status
```

**After (Async)**:
```
API → Worker (fire-and-forget, 10s timeout)
      ↓
      Worker runs independently
      ↓
      Worker uploads to GCS
      ↓
API polls GCS to detect completion
```

### Implementation

#### 1. API Service (`api_service.py`)
- **Fire-and-forget**: Send request to worker with 10s timeout
- **Don't wait**: Catch timeout exceptions (expected behavior)
- **GCS polling**: Check if manifest exists in GCS to detect completion

```python
# Fire request
await client.post(worker_url, json=job_data, timeout=10.0)

# Later, when status is checked:
if job["status"] == "running":
    manifest_blob = bucket.blob(f"scout/jobs/{job_id}/...manifest.json")
    if manifest_blob.exists():
        job["status"] = "completed"
```

#### 2. Worker Service (`worker_service.py`)
- **Robust log streaming**: Handle closed file errors gracefully
- **Better error handling**: Catch ValueError/OSError in streaming threads
- **Completion logging**: Print clear success messages

```python
def _stream(pipe, writer, is_stderr=False):
    try:
        for line in iter(pipe.readline, ""):
            try:
                writer.write(line)
            except (ValueError, OSError):
                break  # File closed, stop streaming
    finally:
        pipe.close()
```

---

## Benefits

### 1. **Reliability**
- Jobs won't be marked failed due to HTTP timeouts
- Worker runs independently without API dependency
- GCS acts as source of truth for completion

### 2. **Scalability**
- API doesn't hold connections open for hours
- Worker can run as long as needed (up to Cloud Run limit)
- Multiple jobs can run concurrently without blocking API

### 3. **Observability**
- Worker logs show complete execution
- GCS contains all outputs
- API status reflects actual completion state

---

## Deployment

This fix is included in the main deployment. Just run:

```bash
./deploy.sh
```

Or manually:

```bash
# Build
gcloud builds submit --tag gcr.io/delvelabs-scout-agent/scout-agent .

# Deploy API
gcloud run deploy scout-agent-api \
    --image gcr.io/delvelabs-scout-agent/scout-agent \
    --region us-central1 \
    --timeout 3600 \
    --set-env-vars "GCS_BUCKET=scout-agent-outputs" \
    --command python --args api_service.py

# Deploy Worker
gcloud run deploy scout-agent-worker \
    --image gcr.io/delvelabs-scout-agent/scout-agent \
    --region us-central1 \
    --timeout 3600 \
    --set-env-vars "GCS_BUCKET=scout-agent-outputs,..." \
    --command python --args worker_service.py
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
    "keywords": "bidirectional links",
    "subreddits": "PKMS",
    "per_query_limit": 2
  }'
```

Response:
```json
{
  "job_id": "abc-123",
  "status": "pending",
  "created_at": "2025-10-02T14:00:00"
}
```

### 2. Poll Status
```bash
JOB_ID="abc-123"
curl "$API_URL/jobs/$JOB_ID"
```

Expected progression:
- `"status": "pending"` → Job queued
- `"status": "running"` → Worker processing
- `"status": "completed"` → GCS manifest detected
- `"gcs_output_path": "gs://..."` → Results available

### 3. Monitor Worker Logs
```bash
gcloud run services logs read scout-agent-worker \
  --region us-central1 \
  --limit 100
```

Look for:
- ✅ No "I/O operation on closed file" errors
- ✅ "Job {job_id} completed successfully"
- ✅ "Output uploaded to: gs://..."
- ✅ "Total files uploaded: N"

### 4. Download Results
```bash
gsutil -m cp -r "gs://scout-agent-outputs/scout/jobs/$JOB_ID" ./results/
```

---

## Troubleshooting

### Job Stuck in "running" Status

**Check worker logs**:
```bash
gcloud run services logs read scout-agent-worker --region us-central1 | grep $JOB_ID
```

**Check if worker is still running**:
- If logs show completion → GCS polling issue
- If logs show errors → Worker crashed
- If no recent logs → Worker may have timed out

**Check GCS manually**:
```bash
gsutil ls gs://scout-agent-outputs/scout/jobs/$JOB_ID/
```

If files exist but status is "running":
- GCS polling path may be wrong
- Check API logs for GCS errors

### Worker Logs Show Errors

**Common errors**:
- `Manifest not found` → Path resolution issue (should be fixed)
- `httpcore.ReadTimeout` → Timeout issue (should be fixed)
- `ValueError: I/O operation` → Log streaming issue (should be fixed)

**Check specific error**:
```bash
gcloud run services logs read scout-agent-worker --region us-central1 | grep ERROR
```

### Job Never Starts

**Check API logs**:
```bash
gcloud run services logs read scout-agent-api --region us-central1 | grep $JOB_ID
```

**Verify worker URL**:
```bash
gcloud run services describe scout-agent-api --region us-central1 --format="value(spec.template.spec.containers[0].env[?(@.name=='WORKER_SERVICE_URL')].value)"
```

---

## Future Improvements

### 1. **Proper Job Queue**
Use Cloud Tasks or Pub/Sub instead of direct HTTP calls:
```
API → Cloud Tasks → Worker
```

Benefits:
- Automatic retries
- Better monitoring
- Rate limiting
- Priority queues

### 2. **Status Callback**
Worker calls API when complete:
```python
# In worker after completion
await client.post(f"{api_url}/jobs/{job_id}/complete", json={
    "status": "completed",
    "gcs_path": "gs://..."
})
```

### 3. **Persistent Storage**
Use Firestore/Redis instead of in-memory `jobs_db`:
```python
# API service
from google.cloud import firestore
db = firestore.Client()
jobs_ref = db.collection('jobs')
```

Benefits:
- Survives API restarts
- Shared across API instances
- Better querying

### 4. **Progress Updates**
Worker streams progress to API:
```python
# Worker
await client.post(f"{api_url}/jobs/{job_id}/progress", json={
    "stage": "scout_collect",
    "progress": 0.3,
    "message": "Collecting Reddit threads..."
})
```

---

## Summary

✅ **Fixed**: Premature "failed" status  
✅ **Fixed**: Log streaming errors  
✅ **Improved**: Job reliability and scalability  
✅ **Added**: GCS-based completion detection  

The system now uses an async, fire-and-forget architecture that's more robust for long-running jobs.
