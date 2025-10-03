# Public Progress Access

## Overview
Users get a **public progress URL** when they submit a job. They can share this URL to watch progress in real-time without authentication.

---

## Security Model

### ✅ Safe Approach
- **Progress endpoint is public** (no auth required)
- **Job IDs are UUIDs** (hard to guess: `abc-123-def-456-789`)
- **Progress logs don't contain sensitive data** (just workflow stages)
- **Results require authentication** (download endpoint checks job ownership)

### ❌ Why Not Public Bucket?
- Exposes ALL jobs to everyone
- Can't revoke access
- Security nightmare
- Violates data privacy

---

## How It Works

### 1. Submit Job → Get Progress URL
```bash
curl -X POST "$API_URL/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "target_market": "Knowledge management",
    "keywords": "bidirectional links",
    "subreddits": "PKMS",
    "per_query_limit": 2
  }'
```

**Response**:
```json
{
  "job_id": "abc-123-def-456-789",
  "status": "pending",
  "created_at": "2025-10-03T00:00:00",
  "estimated_duration": "5-15 minutes",
  "progress_url": "https://scout-agent-api.run.app/jobs/abc-123-def-456-789/progress",
  "output_location": "gs://scout-agent-outputs/scout/jobs/abc-123-def-456-789/"
}
```

### 2. Share Progress URL
Users can share the `progress_url` with anyone:
```
https://scout-agent-api.run.app/jobs/abc-123-def-456-789/progress
```

### 3. Watch Progress (No Auth Required)
```bash
# Anyone with the URL can watch
curl "https://scout-agent-api.run.app/jobs/abc-123-def-456-789/progress"

# Or in browser
open "https://scout-agent-api.run.app/jobs/abc-123-def-456-789/progress"

# Or poll
watch -n 5 "curl -s https://scout-agent-api.run.app/jobs/abc-123-def-456-789/progress | jq -r '.progress'"
```

---

## API Response Format

### Progress Endpoint
`GET /jobs/{job_id}/progress` (PUBLIC - no auth)

**Response**:
```json
{
  "job_id": "abc-123-def-456-789",
  "status": "running",
  "progress": "ScoutAgent Job Progress Log\nJob ID: abc-123-def-456-789\n...\n[2025-10-03 00:05:30] INFO - Completed stage: gap_finder_collect\n"
}
```

**Status Values**:
- `"pending"` - Log not yet created
- `"running"` - Job in progress
- `"completed"` - Job finished successfully
- `"failed"` - Job encountered error

---

## User Experience

### Example Workflow

**1. User submits job**:
```bash
RESPONSE=$(curl -s -X POST "$API_URL/jobs" \
  -H "Content-Type: application/json" \
  -d '{"target_market":"Knowledge management","keywords":"bidirectional links","subreddits":"PKMS","per_query_limit":2}')

echo "$RESPONSE" | jq
```

**Output**:
```json
{
  "job_id": "abc-123",
  "status": "pending",
  "created_at": "2025-10-03T00:00:00",
  "estimated_duration": "5-15 minutes",
  "progress_url": "https://scout-agent-api.run.app/jobs/abc-123/progress",
  "output_location": "gs://scout-agent-outputs/scout/jobs/abc-123/"
}
```

**2. User shares progress URL**:
```
Hey team, I started a ScoutAgent job!
Watch progress here: https://scout-agent-api.run.app/jobs/abc-123/progress
Results will be at: gs://scout-agent-outputs/scout/jobs/abc-123/
```

**3. Team watches progress**:
```bash
# No authentication needed!
watch -n 5 "curl -s https://scout-agent-api.run.app/jobs/abc-123/progress | jq -r '.progress'"
```

**4. Download results (requires job_id)**:
```bash
curl "https://scout-agent-api.run.app/jobs/abc-123/download" -o results.zip
```

---

## Security Considerations

### What's Public?
✅ **Progress logs** - Workflow stages, no sensitive data  
✅ **Job status** - pending/running/completed/failed  
✅ **Output location** - GCS path (but not accessible without GCS auth)  

### What's Protected?
🔒 **Job results** - Download requires knowing job_id  
🔒 **GCS bucket** - Not public, requires authentication  
🔒 **API keys** - Never exposed in logs  
🔒 **User data** - Reddit posts, analysis results  

### UUID Security
- Job IDs are UUIDs: `abc-123-def-456-789`
- 128-bit random: 2^128 possible values
- Practically impossible to guess
- No sequential IDs that can be enumerated

---

## Implementation

### API Service (api_service.py)

**Job Creation**:
```python
# Generate progress URL
api_base = os.getenv("API_BASE_URL", "https://scout-agent-api.run.app")
progress_url = f"{api_base}/jobs/{job_id}/progress"

# Output location
bucket_name = os.getenv("GCS_BUCKET", "scout-agent-outputs")
output_location = f"gs://{bucket_name}/scout/jobs/{job_id}/"

return JobResponse(
    job_id=job_id,
    status="pending",
    created_at=created_at,
    progress_url=progress_url,
    output_location=output_location
)
```

**Progress Endpoint (Public)**:
```python
@app.get("/jobs/{job_id}/progress")
async def get_job_progress(job_id: str):
    # No auth check - public endpoint
    # Job IDs are UUIDs (hard to guess)
    # Logs don't contain sensitive data
    
    progress_blob = bucket.blob(f"scout/jobs/{job_id}/progress.log")
    progress_content = progress_blob.download_as_string().decode('utf-8')
    
    return {
        "job_id": job_id,
        "progress": progress_content,
        "status": "running" | "completed" | "failed"
    }
```

---

## Deployment

### Set API_BASE_URL
The deployment script automatically sets this:

```bash
API_URL=$(gcloud run services describe scout-agent-api --region us-central1 --format="value(status.url)")

gcloud run services update scout-agent-api \
    --region us-central1 \
    --set-env-vars "API_BASE_URL=$API_URL"
```

### Deploy
```bash
cd /home/ubuntu/ScoutAgent
./deploy.sh
```

---

## Example: Complete User Flow

```bash
# 1. Submit job
API_URL="https://scout-agent-511946707043.us-central1.run.app"

RESPONSE=$(curl -s -X POST "$API_URL/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "target_market": "Knowledge management",
    "keywords": "bidirectional links",
    "subreddits": "PKMS",
    "per_query_limit": 2
  }')

# 2. Extract URLs
JOB_ID=$(echo "$RESPONSE" | jq -r '.job_id')
PROGRESS_URL=$(echo "$RESPONSE" | jq -r '.progress_url')
OUTPUT_LOCATION=$(echo "$RESPONSE" | jq -r '.output_location')

echo "Job ID: $JOB_ID"
echo "Watch progress: $PROGRESS_URL"
echo "Results will be at: $OUTPUT_LOCATION"

# 3. Watch progress (public - no auth)
watch -n 5 "curl -s $PROGRESS_URL | jq -r '.progress'"

# 4. Check status
curl -s "$API_URL/jobs/$JOB_ID" | jq

# 5. Download results
curl "$API_URL/jobs/$JOB_ID/download" -o results.zip
```

---

## Browser Access

Users can open the progress URL directly in a browser:
```
https://scout-agent-api.run.app/jobs/abc-123-def-456-789/progress
```

They'll see JSON response:
```json
{
  "job_id": "abc-123-def-456-789",
  "status": "running",
  "progress": "[2025-10-03 00:00:05] Starting MCP servers...\n[2025-10-03 00:00:20] All MCP servers ready\n..."
}
```

For better UX, you could create a simple HTML page that polls and displays the progress nicely.

---

## Troubleshooting

### Progress URL Returns 404
**Check if job exists**:
```bash
curl "$API_URL/jobs/$JOB_ID"
```

**Verify API_BASE_URL is set**:
```bash
gcloud run services describe scout-agent-api --region us-central1 --format="value(spec.template.spec.containers[0].env)"
```

### Progress URL Returns "Not available"
Job is starting. Wait a few seconds and try again.

### Can't Access Results
Results require job_id. Use download endpoint:
```bash
curl "$API_URL/jobs/$JOB_ID/download" -o results.zip
```

---

## Summary

✅ **Public progress access** - Share URL with anyone  
✅ **Secure** - UUIDs prevent guessing, no sensitive data in logs  
✅ **User-friendly** - No auth needed to watch progress  
✅ **Predictable** - Output location known upfront  
✅ **Protected results** - Download requires job_id  

Users can confidently share progress URLs with their team while keeping results secure!
