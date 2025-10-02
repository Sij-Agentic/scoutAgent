# ScoutAgent Cloud Run - Quick Reference

## Deploy
```bash
cd /home/ubuntu/ScoutAgent
./deploy.sh

# Then set API keys:
gcloud run services update scout-agent-worker --region us-central1 \
  --set-env-vars "SCOUT_OPENAI_API_KEY=sk-...,SCOUT_ANTHROPIC_API_KEY=sk-ant-...,SCOUT_REDDIT_CLIENT_ID=...,SCOUT_REDDIT_CLIENT_SECRET=...,SCOUT_REDDIT_USER_AGENT=scout-agent/1.0"
```

## Submit Job
```bash
API_URL="https://scout-agent-511946707043.us-central1.run.app"

curl -X POST "$API_URL/jobs" -H "Content-Type: application/json" -d '{
  "target_market": "Knowledge management tools",
  "keywords": "bidirectional links,markdown sync",
  "subreddits": "PKMS,productivity",
  "per_query_limit": 5
}'
```

## Check Status
```bash
JOB_ID="<from_submission>"
curl "$API_URL/jobs/$JOB_ID"
```

## View Logs
```bash
# Clean logs (default)
gcloud run services logs read scout-agent-worker --region us-central1 --limit 100

# Search for errors
gcloud run services logs read scout-agent-worker --region us-central1 | grep -i error

# Enable verbose MCP logs (debugging only)
gcloud run services update scout-agent-worker --region us-central1 --set-env-vars "VERBOSE_MCP_LOGS=true"
```

## Download Results
```bash
JOB_ID="<job_id>"
gsutil -m cp -r "gs://scout-agent-outputs/scout/jobs/$JOB_ID" ./results/
```

## Verify Configuration
```bash
# Check timeout (should be 3600)
gcloud run services describe scout-agent-worker --region us-central1 --format="value(spec.template.spec.timeoutSeconds)"

# Check env vars
gcloud run services describe scout-agent-worker --region us-central1 --format="value(spec.template.spec.containers[0].env)"
```

## Troubleshooting

### Job Fails Immediately
- Check API keys are set
- Check worker logs for errors

### Job Times Out
- Check if >1 hour runtime (Cloud Run limit)
- Reduce `per_query_limit`

### Can't Find Errors
- Logs are clean by default
- Search: `grep -i "error\|failed\|exception"`
- Enable verbose: `VERBOSE_MCP_LOGS=true`

### Status Stuck on "Running"
- Check GCS: `gsutil ls gs://scout-agent-outputs/scout/jobs/$JOB_ID/`
- If files exist, job completed (status will update on next poll)

## All Fixes Applied
✅ SSE timeout (4 min) → 3600s  
✅ Cloud Run timeout → 3600s  
✅ Process wait timeout (15 min) → 3600s  
✅ Manifest path resolution  
✅ Fire-and-forget job processing  
✅ MCP log verbosity control  
✅ Better error reporting  

## Files Changed
- `scout_agent/agents/scout.py` - MCP timeout
- `api_service.py` - Fire-and-forget + GCS polling
- `worker_service.py` - Process timeout + log control
- `Dockerfile` - Data directory
- `deploy.sh` - All env vars and timeouts
