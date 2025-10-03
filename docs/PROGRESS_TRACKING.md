# Real-Time Progress Tracking

## Overview
Users can now **watch job progress in real-time** by reading a live progress log stored in GCS. No more guessing if the job is running or stuck!

---

## How It Works

```
1. Job starts → Creates progress.log in GCS
2. Worker streams important logs to progress.log
3. User polls /jobs/{job_id}/progress
4. See real-time updates as job runs
5. Final output location is known upfront
```

---

## Features

### ✅ Predictable Output Location
Output is always at: `gs://scout-agent-outputs/scout/jobs/{job_id}/`

### ✅ Real-Time Progress Log
Streams to: `gs://scout-agent-outputs/scout/jobs/{job_id}/progress.log`

### ✅ Key Events Logged
- MCP server startup
- Workflow stages (scout_collect, gap_finder_collect, etc.)
- Important INFO/WARNING/ERROR messages
- Upload progress
- Completion status

### ✅ User-Friendly API
- `GET /jobs/{job_id}/progress` - View progress log
- `GET /jobs/{job_id}` - Check status
- `GET /jobs/{job_id}/download` - Download results

---

## API Usage

### 1. Submit Job
```bash
API_URL="https://scout-agent-api.run.app"

RESPONSE=$(curl -s -X POST "$API_URL/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "target_market": "Knowledge management tools",
    "keywords": "bidirectional links",
    "subreddits": "PKMS",
    "per_query_limit": 2
  }')

JOB_ID=$(echo $RESPONSE | jq -r '.job_id')
echo "Job ID: $JOB_ID"
echo "Progress: $API_URL/jobs/$JOB_ID/progress"
echo "Output will be at: gs://scout-agent-outputs/scout/jobs/$JOB_ID/"
```

### 2. Watch Progress (Real-Time)
```bash
# Poll progress log every 5 seconds
watch -n 5 "curl -s $API_URL/jobs/$JOB_ID/progress | jq -r '.progress'"
```

**Example Output**:
```
ScoutAgent Job Progress Log
Job ID: abc-123-def
Started: 2025-10-03T00:00:00
Target Market: Knowledge management tools
Keywords: bidirectional links
Subreddits: PKMS
Per Query Limit: 2

Output will be available at: gs://scout-agent-outputs/scout/jobs/abc-123-def/

================================================================================
PROGRESS LOG
================================================================================

[2025-10-03 00:00:05] Starting MCP servers...
[2025-10-03 00:00:10] Starting MCP server: gap_finder_tools on port 8000
[2025-10-03 00:00:12] Starting MCP server: reddit_api on port 8001
[2025-10-03 00:00:14] Starting MCP server: research_tools on port 8002
[2025-10-03 00:00:16] Starting MCP server: web_search on port 8004
[2025-10-03 00:00:20] All MCP servers ready
[2025-10-03 00:00:20] Starting ScoutAgent workflow...
[2025-10-03 00:00:25] INFO - Starting Scout agent workflow
[2025-10-03 00:00:30] INFO - Executing stage: scout_collect
[2025-10-03 00:02:15] INFO - Completed stage: scout_collect
[2025-10-03 00:02:20] INFO - Starting Gap Finder agent
[2025-10-03 00:05:30] INFO - Completed stage: gap_finder_collect
[2025-10-03 00:10:45] ScoutAgent workflow completed successfully
[2025-10-03 00:10:50] Uploading results to GCS...
[2025-10-03 00:11:00] Upload complete: 42 files
[2025-10-03 00:11:00] Results available at: gs://scout-agent-outputs/scout/jobs/abc-123-def/
[2025-10-03 00:11:00] Job completed successfully!
```

### 3. Check Status
```bash
curl "$API_URL/jobs/$JOB_ID"
```

### 4. Download Results
```bash
curl "$API_URL/jobs/$JOB_ID/download" -o results.zip
```

---

## Progress Log Format

### Header
```
ScoutAgent Job Progress Log
Job ID: {job_id}
Started: {timestamp}
Target Market: {target_market}
Keywords: {keywords}
Subreddits: {subreddits}
Per Query Limit: {limit}

Output will be available at: gs://{bucket}/{prefix}
```

### Log Entries
```
[YYYY-MM-DD HH:MM:SS] {message}
```

### Key Events
- **MCP Startup**: "Starting MCP server: {name} on port {port}"
- **Workflow Start**: "Starting ScoutAgent workflow..."
- **Stage Progress**: "INFO - Executing stage: {stage_name}"
- **Stage Complete**: "INFO - Completed stage: {stage_name}"
- **Upload**: "Uploading results to GCS..."
- **Completion**: "Job completed successfully!"
- **Errors**: "ERROR: {error_message}"

---

## Implementation Details

### Worker (worker_service.py)

**1. Create Progress Log**:
```python
progress_log_path = f"scout/jobs/{job_id}/progress.log"
progress_blob = bucket.blob(progress_log_path)
progress_blob.upload_from_string(initial_log)
```

**2. Log Helper Function**:
```python
def log_to_gcs(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    current_content = progress_blob.download_as_string().decode('utf-8')
    updated_content = current_content + log_entry
    progress_blob.upload_from_string(updated_content)
```

**3. Stream Important Logs**:
```python
# In stdout/stderr streaming
if any(keyword in line_lower for keyword in ['info', 'starting', 'completed', 'error', 'warning', 'stage']):
    log_to_gcs(line.strip())
```

### API (api_service.py)

**Progress Endpoint**:
```python
@app.get("/jobs/{job_id}/progress")
async def get_job_progress(job_id: str):
    progress_blob = bucket.blob(f"scout/jobs/{job_id}/progress.log")
    progress_content = progress_blob.download_as_string().decode('utf-8')
    return {"progress": progress_content}
```

---

## Benefits

### For Users
✅ **See what's happening** - Real-time progress updates  
✅ **Know where output is** - Location specified upfront  
✅ **Debug easily** - Error messages in progress log  
✅ **No surprises** - Watch each stage complete  

### For Developers
✅ **Easy debugging** - All logs in one place  
✅ **No Cloud Run access needed** - Users can self-serve  
✅ **Predictable paths** - Consistent GCS structure  
✅ **Audit trail** - Complete log of job execution  

---

## Example: Complete Workflow

```bash
# 1. Submit job
JOB_ID=$(curl -s -X POST "$API_URL/jobs" \
  -H "Content-Type: application/json" \
  -d '{"target_market":"Knowledge management","keywords":"bidirectional links","subreddits":"PKMS","per_query_limit":2}' \
  | jq -r '.job_id')

echo "Job ID: $JOB_ID"
echo "Watch progress: watch -n 5 \"curl -s $API_URL/jobs/$JOB_ID/progress | jq -r '.progress'\""

# 2. Watch progress in real-time
watch -n 5 "curl -s $API_URL/jobs/$JOB_ID/progress | jq -r '.progress'"

# 3. Check status
curl "$API_URL/jobs/$JOB_ID" | jq

# 4. Download when complete
curl "$API_URL/jobs/$JOB_ID/download" -o results.zip

# 5. Or access directly from GCS
gsutil -m cp -r "gs://scout-agent-outputs/scout/jobs/$JOB_ID" ./results/
```

---

## Troubleshooting

### Progress Log Not Updating
**Check if worker is running**:
```bash
gcloud run services logs read scout-agent-worker --region us-central1 | grep $JOB_ID
```

**Check GCS directly**:
```bash
gsutil cat gs://scout-agent-outputs/scout/jobs/$JOB_ID/progress.log
```

### Progress Shows Error
**Read full error context**:
```bash
curl "$API_URL/jobs/$JOB_ID/progress" | jq -r '.progress' | grep -A 5 ERROR
```

**Check worker logs**:
```bash
gcloud run services logs read scout-agent-worker --region us-central1 | grep -i error
```

### Job Stuck
**Last progress entry**:
```bash
curl "$API_URL/jobs/$JOB_ID/progress" | jq -r '.progress' | tail -20
```

If no updates for >5 minutes, job may have crashed. Check worker logs.

---

## Deployment

```bash
cd /home/ubuntu/ScoutAgent
./force_deploy.sh
```

Set API keys after deployment (shown in output).

---

## Summary

✅ **Real-time progress tracking** - Watch jobs as they run  
✅ **Predictable output location** - Known upfront  
✅ **User-friendly API** - No GCS access needed  
✅ **Complete audit trail** - All logs in one place  
✅ **Easy debugging** - Errors visible immediately  

Users can now confidently submit jobs and watch them progress in real-time!
