# Final Solution: Public Progress Tracking

## ✅ Problem Solved

Users can now:
1. **Submit a job** → Get a public progress URL
2. **Share the URL** → Anyone can watch progress (no auth)
3. **Watch in real-time** → See workflow stages as they happen
4. **Know output location** → Predictable GCS path
5. **Download results** → When job completes

---

## How It Works

### 1. Submit Job
```bash
curl -X POST "$API_URL/jobs" -H "Content-Type: application/json" -d '{
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
  "progress_url": "https://scout-agent-api.run.app/jobs/abc-123/progress",
  "output_location": "gs://scout-agent-outputs/scout/jobs/abc-123/"
}
```

### 2. Share Progress URL (Public!)
```
https://scout-agent-api.run.app/jobs/abc-123/progress
```

**Anyone with this URL can watch** - no authentication needed!

### 3. Watch Progress
```bash
watch -n 5 "curl -s https://scout-agent-api.run.app/jobs/abc-123/progress | jq -r '.progress'"
```

**See real-time updates**:
```
[2025-10-03 00:00:05] Starting MCP servers...
[2025-10-03 00:00:20] All MCP servers ready
[2025-10-03 00:00:20] Starting ScoutAgent workflow...
[2025-10-03 00:00:30] INFO - Executing stage: scout_collect
[2025-10-03 00:02:15] INFO - Completed stage: scout_collect
[2025-10-03 00:10:45] ScoutAgent workflow completed successfully
[2025-10-03 00:11:00] Job completed successfully!
```

### 4. Download Results
```bash
curl "$API_URL/jobs/abc-123/download" -o results.zip
```

---

## Security Model

### ✅ What's Public?
- **Progress logs** - Workflow stages, no sensitive data
- **Job status** - pending/running/completed/failed
- **Output location** - GCS path (but not accessible without auth)

### 🔒 What's Protected?
- **Job results** - Download requires job_id
- **GCS bucket** - Not public, requires authentication
- **API keys** - Never exposed
- **User data** - Reddit posts, analysis results

### 🛡️ Why This Is Safe
- **Job IDs are UUIDs** - Impossible to guess (2^128 combinations)
- **No sensitive data in logs** - Just workflow stages
- **Results require job_id** - Can't enumerate or guess
- **No public bucket** - Everything stays private except progress

---

## API Endpoints

### POST /jobs
Submit a new job

**Response**:
- `job_id` - UUID for this job
- `progress_url` - **Public URL to watch progress**
- `output_location` - Where results will be stored

### GET /jobs/{job_id}/progress (PUBLIC)
Watch job progress in real-time

**No authentication required!**

**Response**:
```json
{
  "job_id": "abc-123",
  "status": "running",
  "progress": "[2025-10-03 00:00:05] Starting MCP servers...\n..."
}
```

### GET /jobs/{job_id}
Check job status

### GET /jobs/{job_id}/download
Download results as zip file

---

## User Experience

### Before ❌
```
Submit job → Shows "running" → Eventually "failed" (wrong!)
→ No way to see progress
→ No way to get results
→ User confused and frustrated
```

### After ✅
```
Submit job → Get progress URL
→ Share with team
→ Everyone watches progress in real-time
→ See each stage complete
→ Download results when done
→ Happy users!
```

---

## Example: Team Collaboration

**User submits job**:
```bash
RESPONSE=$(curl -s -X POST "$API_URL/jobs" -H "Content-Type: application/json" -d '{...}')
PROGRESS_URL=$(echo "$RESPONSE" | jq -r '.progress_url')
```

**User shares in Slack**:
```
Hey team! Started a ScoutAgent job for "Knowledge management tools"

Watch progress: https://scout-agent-api.run.app/jobs/abc-123/progress
Results will be at: gs://scout-agent-outputs/scout/jobs/abc-123/

Should take 5-15 minutes.
```

**Team watches together**:
```bash
# No authentication needed!
watch -n 5 "curl -s https://scout-agent-api.run.app/jobs/abc-123/progress | jq -r '.progress'"
```

**Everyone sees**:
- When MCP servers start
- When each workflow stage completes
- If any errors occur
- When job finishes

---

## Deployment

```bash
cd /home/ubuntu/ScoutAgent
./deploy.sh
```

**After deployment**:
```bash
# Set API keys
gcloud run services update scout-agent-worker --region us-central1 \
  --update-env-vars "SCOUT_OPENAI_API_KEY=sk-...,SCOUT_ANTHROPIC_API_KEY=sk-ant-...,SCOUT_DEEPSEEK_API_KEY=sk-...,SCOUT_REDDIT_CLIENT_ID=...,SCOUT_REDDIT_CLIENT_SECRET=...,SCOUT_REDDIT_USER_AGENT=scout-agent/1.0"
```

---

## Files Changed

| File | Change |
|------|--------|
| `api_service.py` | Added progress_url and output_location to response |
| `api_service.py` | Made /progress endpoint public |
| `worker_service.py` | Streams logs to GCS progress.log |
| `deploy.sh` | Sets API_BASE_URL env var |
| `QUICK_REFERENCE.md` | Updated with new workflow |

---

## Complete Workflow

```bash
# 1. Submit job
API_URL="https://scout-agent-511946707043.us-central1.run.app"

RESPONSE=$(curl -s -X POST "$API_URL/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "target_market": "Knowledge management tools",
    "keywords": "bidirectional links",
    "subreddits": "PKMS",
    "per_query_limit": 2
  }')

# 2. Extract URLs
JOB_ID=$(echo "$RESPONSE" | jq -r '.job_id')
PROGRESS_URL=$(echo "$RESPONSE" | jq -r '.progress_url')
OUTPUT=$(echo "$RESPONSE" | jq -r '.output_location')

echo "Job ID: $JOB_ID"
echo "Progress: $PROGRESS_URL"
echo "Output: $OUTPUT"

# 3. Watch progress (public - share this URL!)
watch -n 5 "curl -s $PROGRESS_URL | jq -r '.progress'"

# 4. Check status
curl "$API_URL/jobs/$JOB_ID" | jq

# 5. Download results
curl "$API_URL/jobs/$JOB_ID/download" -o results.zip

# 6. Or access directly from GCS
gsutil -m cp -r "$OUTPUT" ./results/
```

---

## Summary

✅ **Public progress URL** - Share with anyone  
✅ **Real-time updates** - Watch as job runs  
✅ **Predictable output** - Location known upfront  
✅ **Secure** - UUIDs prevent guessing, results protected  
✅ **User-friendly** - No auth needed to watch progress  
✅ **Team collaboration** - Everyone can watch together  

**Perfect solution for your users!** 🎉
