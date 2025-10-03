# User Experience - Complete Workflow

## What Users See

### 1. Submit Job
```bash
python3 test_cloudrun.py
```

**Output**:
```
🚀 ScoutAgent Cloud Run API Test
API URL: https://scout-agent-511946707043.us-central1.run.app

🧪 Testing job creation...

✅ Job created successfully!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 Job ID: abc-123-def-456-789
⏱️  Estimated Duration: 5-15 minutes

📊 Watch Progress (Public - Shareable!):
   https://scout-agent-api.run.app/jobs/abc-123/progress

📁 Final Output Location:
   gs://scout-agent-outputs/scout/jobs/abc-123/
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 Tip: Share the progress URL with your team to watch together!
💡 Progress log available for 24 hours

⏳ Waiting for job to complete...
💡 Watch live: watch -n 5 "curl -s https://scout-agent-api.run.app/jobs/abc-123/progress | jq -r '.progress'"
```

### 2. Watch Progress
Users can watch in real-time:
```bash
watch -n 5 "curl -s https://scout-agent-api.run.app/jobs/abc-123/progress | jq -r '.progress'"
```

**They see**:
```
ScoutAgent Job Progress Log
Job ID: abc-123-def-456-789
Started: 2025-10-03T00:00:00
Target Market: Knowledge management tools
Keywords: bidirectional links,markdown sync
Subreddits: PKMS,productivity

Output will be available at: gs://scout-agent-outputs/scout/jobs/abc-123/

================================================================================
PROGRESS LOG
================================================================================

[2025-10-03 00:00:05] Starting MCP servers...
[2025-10-03 00:00:20] All MCP servers ready
[2025-10-03 00:00:20] Starting ScoutAgent workflow...
[2025-10-03 00:00:30] INFO - Executing stage: scout_collect
[2025-10-03 00:02:15] INFO - Completed stage: scout_collect
[2025-10-03 00:02:20] INFO - Starting Gap Finder agent
[2025-10-03 00:05:30] INFO - Completed stage: gap_finder_collect
[2025-10-03 00:10:45] ScoutAgent workflow completed successfully
[2025-10-03 00:11:00] Upload complete: 42 files
[2025-10-03 00:11:00] Results available at: gs://scout-agent-outputs/scout/jobs/abc-123/
[2025-10-03 00:11:00] Job completed successfully!
```

### 3. Job Completes
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎉 Job completed successfully!

📁 Download Results:
   curl "https://scout-agent-api.run.app/jobs/abc-123/download" -o results.zip

📂 Or access directly from GCS:
   gsutil -m cp -r "gs://scout-agent-outputs/scout/jobs/abc-123/" ./results/

📊 View Progress Log:
   https://scout-agent-api.run.app/jobs/abc-123/progress
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Key Information Provided

### ✅ At Job Submission
1. **Job ID** - Unique identifier
2. **Estimated Duration** - How long it will take
3. **Progress URL** - Public, shareable link to watch progress
4. **Output Location** - Where results will be stored
5. **Tips** - How to share and watch

### ✅ During Execution
1. **Real-time progress** - See each stage complete
2. **MCP server status** - Know when tools are ready
3. **Workflow stages** - Track scout_collect, gap_finder, etc.
4. **Errors** - Immediate visibility if something fails

### ✅ At Completion
1. **Success confirmation** - Clear completion message
2. **Download command** - Copy-paste ready
3. **GCS path** - Direct access option
4. **Progress log link** - Review what happened

---

## Progress Log Availability

### 24-Hour Access
- Progress log stays in GCS for 24 hours
- Users can review it anytime
- Share with team members who missed it
- Debug issues after the fact

### After 24 Hours
- Progress log can be manually deleted or kept
- Results stay in GCS indefinitely (until manually deleted)
- Job status still available via API

---

## Sharing with Team

### Example Slack Message
```
Hey team! I started a ScoutAgent job for "Knowledge management tools"

📊 Watch progress (live updates):
https://scout-agent-api.run.app/jobs/abc-123/progress

📁 Results will be at:
gs://scout-agent-outputs/scout/jobs/abc-123/

Should take 5-15 minutes. Progress log available for 24 hours!
```

### Team Can
- ✅ Watch progress together (no auth needed)
- ✅ See when job completes
- ✅ Access results (with GCS permissions)
- ✅ Review progress log later

---

## API Response Structure

### Job Creation Response
```json
{
  "job_id": "abc-123-def-456-789",
  "status": "pending",
  "created_at": "2025-10-03T00:00:00.123456",
  "estimated_duration": "5-15 minutes",
  "progress_url": "https://scout-agent-api.run.app/jobs/abc-123/progress",
  "output_location": "gs://scout-agent-outputs/scout/jobs/abc-123/"
}
```

### Progress Response (Public)
```json
{
  "job_id": "abc-123-def-456-789",
  "status": "running",
  "progress": "[2025-10-03 00:00:05] Starting MCP servers...\n..."
}
```

### Status Response
```json
{
  "job_id": "abc-123-def-456-789",
  "status": "completed",
  "created_at": "2025-10-03T00:00:00.123456",
  "completed_at": "2025-10-03T00:15:00.123456",
  "gcs_output_path": "gs://scout-agent-outputs/scout/jobs/abc-123/",
  "error_message": null
}
```

---

## Error Handling

### If Job Fails
```
❌ Job failed
Error: Job abc-123 timed out after 3600s

📊 Check progress log for details:
   https://scout-agent-api.run.app/jobs/abc-123/progress
```

Users can:
1. Check progress log to see where it failed
2. See error messages in context
3. Retry with adjusted parameters

---

## Complete Example

```bash
# 1. Submit job
python3 test_cloudrun.py

# Output shows:
# - Job ID
# - Progress URL (public, shareable)
# - Output location (predictable)

# 2. Share with team
# Send progress URL in Slack/email

# 3. Watch together
watch -n 5 "curl -s https://scout-agent-api.run.app/jobs/abc-123/progress | jq -r '.progress'"

# 4. Download results when complete
curl "https://scout-agent-api.run.app/jobs/abc-123/download" -o results.zip

# Or access from GCS
gsutil -m cp -r "gs://scout-agent-outputs/scout/jobs/abc-123/" ./results/
```

---

## Summary

### Users Get
✅ **Clear job information** - ID, duration, URLs  
✅ **Public progress URL** - Share with team  
✅ **Predictable output location** - Known upfront  
✅ **Real-time updates** - Watch as job runs  
✅ **24-hour access** - Review progress anytime  
✅ **Easy download** - Copy-paste commands  
✅ **Final output location** - In status response  

### Perfect User Experience
1. Submit → Get all URLs immediately
2. Share → Team watches together
3. Complete → Download with one command
4. Review → Progress log available for 24 hours

**Everything they need, clearly displayed!** 🎉
