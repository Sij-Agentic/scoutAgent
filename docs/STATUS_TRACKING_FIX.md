# Status Tracking Fix - GCS-Based Job Completion

## Problem
- Jobs show "failed" even though they're still running
- Users can't tell when jobs actually complete
- No way to access final output without GCS access

## Solution: GCS Status File

### How It Works

```
1. User submits job → API returns job_id
2. API fires request to worker (doesn't wait)
3. Worker processes job independently
4. Worker writes job_status.json to GCS when done
5. User polls /jobs/{job_id} → API checks GCS status file
6. Status updates to "completed" or "failed" with details
7. User downloads results via /jobs/{job_id}/download
```

---

## Changes Made

### 1. Worker Writes Status File (worker_service.py)

**On Success**:
```python
status_data = {
    "job_id": request.job_id,
    "status": "completed",
    "completed_at": datetime.utcnow().isoformat(),
    "gcs_output_path": gcs_output_path,
    "files_uploaded": len(uploaded_files),
    "error": None
}
status_blob = bucket.blob(f"{gcs_prefix}job_status.json")
status_blob.upload_from_string(json.dumps(status_data, indent=2))
```

**On Failure**:
```python
status_data = {
    "job_id": request.job_id,
    "status": "failed",
    "completed_at": datetime.utcnow().isoformat(),
    "error": error_msg,
    "error_type": "timeout" | "exception"
}
```

### 2. API Checks Status File (api_service.py)

```python
# Check for status file in GCS
status_blob = bucket.blob(f"scout/jobs/{job_id}/job_status.json")
if status_blob.exists():
    status_data = json.loads(status_blob.download_as_string())
    job["status"] = status_data.get("status", "completed")
    job["completed_at"] = status_data.get("completed_at")
    job["gcs_output_path"] = status_data.get("gcs_output_path")
    if status_data.get("error"):
        job["error_message"] = status_data.get("error")
```

### 3. Download Endpoint (api_service.py)

**New endpoint**: `GET /jobs/{job_id}/download`

Returns a zip file with all job outputs:
- Manifest files
- Reddit data
- Analysis results
- Logs

---

## API Usage

### 1. Submit Job
```bash
curl -X POST "https://scout-agent-api.run.app/jobs" \
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
  "job_id": "abc-123-def",
  "status": "pending",
  "created_at": "2025-10-03T00:00:00",
  "estimated_duration": "5-15 minutes"
}
```

### 2. Check Status
```bash
curl "https://scout-agent-api.run.app/jobs/abc-123-def"
```

Response (running):
```json
{
  "job_id": "abc-123-def",
  "status": "running",
  "created_at": "2025-10-03T00:00:00",
  "completed_at": null,
  "gcs_output_path": null,
  "error_message": null
}
```

Response (completed):
```json
{
  "job_id": "abc-123-def",
  "status": "completed",
  "created_at": "2025-10-03T00:00:00",
  "completed_at": "2025-10-03T00:15:00",
  "gcs_output_path": "gs://scout-agent-outputs/scout/jobs/abc-123-def/",
  "error_message": null
}
```

Response (failed):
```json
{
  "job_id": "abc-123-def",
  "status": "failed",
  "created_at": "2025-10-03T00:00:00",
  "completed_at": "2025-10-03T00:05:00",
  "gcs_output_path": null,
  "error_message": "Job abc-123-def timed out after 3600s"
}
```

### 3. Download Results
```bash
curl "https://scout-agent-api.run.app/jobs/abc-123-def/download" \
  -o scout_results.zip
```

Returns: `scout_job_abc-123-def.zip` containing all outputs

---

## Benefits

### ✅ Accurate Status
- No more false "failed" status
- Real-time completion detection
- Error messages when jobs actually fail

### ✅ User-Friendly
- Poll `/jobs/{job_id}` to check status
- Download results directly via API
- No need for GCS access or gsutil

### ✅ Reliable
- Status persisted in GCS (survives API restarts)
- Works with fire-and-forget architecture
- Handles timeouts and exceptions

---

## Deployment

```bash
cd /home/ubuntu/ScoutAgent
./force_deploy.sh
```

After deployment, set API keys:
```bash
gcloud run services update scout-agent-worker --region us-central1 \
  --update-env-vars "SCOUT_OPENAI_API_KEY=sk-...,SCOUT_ANTHROPIC_API_KEY=sk-ant-...,SCOUT_DEEPSEEK_API_KEY=sk-...,SCOUT_REDDIT_CLIENT_ID=...,SCOUT_REDDIT_CLIENT_SECRET=...,SCOUT_REDDIT_USER_AGENT=scout-agent/1.0"
```

---

## Testing

### 1. Submit Job
```bash
API_URL="https://scout-agent-511946707043.us-central1.run.app"

JOB_RESPONSE=$(curl -s -X POST "$API_URL/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "target_market": "Knowledge management",
    "keywords": "bidirectional links",
    "subreddits": "PKMS",
    "per_query_limit": 2
  }')

JOB_ID=$(echo $JOB_RESPONSE | jq -r '.job_id')
echo "Job ID: $JOB_ID"
```

### 2. Poll Status
```bash
watch -n 10 "curl -s $API_URL/jobs/$JOB_ID | jq"
```

Watch for:
- `"status": "pending"` → Job queued
- `"status": "running"` → Worker processing
- `"status": "completed"` → Job done!
- `"gcs_output_path"` appears when complete

### 3. Download Results
```bash
curl "$API_URL/jobs/$JOB_ID/download" -o results.zip
unzip results.zip
ls -la
```

---

## Status File Format

**Location**: `gs://scout-agent-outputs/scout/jobs/{job_id}/job_status.json`

**Success**:
```json
{
  "job_id": "abc-123-def",
  "status": "completed",
  "completed_at": "2025-10-03T00:15:00.123456",
  "gcs_output_path": "gs://scout-agent-outputs/scout/jobs/abc-123-def/",
  "files_uploaded": 42,
  "error": null
}
```

**Failure**:
```json
{
  "job_id": "abc-123-def",
  "status": "failed",
  "completed_at": "2025-10-03T00:05:00.123456",
  "error": "Job abc-123-def timed out after 3600s",
  "error_type": "timeout"
}
```

---

## Troubleshooting

### Status Stuck on "running"
**Check worker logs**:
```bash
gcloud run services logs read scout-agent-worker --region us-central1 | grep $JOB_ID
```

**Check if status file exists**:
```bash
gsutil cat gs://scout-agent-outputs/scout/jobs/$JOB_ID/job_status.json
```

### Download Fails
**Error**: "Job is not completed yet"
- Wait for status to change to "completed"
- Check status endpoint first

**Error**: "No output path found"
- Job may have failed before writing outputs
- Check error_message in status response

### Status Never Updates
**Possible causes**:
1. Worker crashed before writing status file
2. GCS permissions issue
3. API can't read from GCS

**Fix**:
- Check worker logs for errors
- Verify GCS bucket permissions
- Check API service has Storage Object Viewer role

---

## Summary

✅ **Worker writes status file** to GCS on completion/failure  
✅ **API reads status file** to update job status  
✅ **Download endpoint** provides easy access to results  
✅ **No more false failures** - accurate status tracking  
✅ **User-friendly** - poll and download via API  

Deploy with `./force_deploy.sh` and test!
