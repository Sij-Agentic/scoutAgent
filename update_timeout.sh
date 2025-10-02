#!/bin/bash

# Quick manual timeout update for Cloud Run services
# Extracted from your API URL: scout-agent-511946707043.us-central1.run.app

PROJECT_ID="delvelabs-scout-agent"
REGION="us-central1"

echo "🔧 Updating Cloud Run timeout settings"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo ""

# Update API Service
echo "⏱️  Updating scout-agent-api timeout..."
gcloud run services update scout-agent-api \
    --project=$PROJECT_ID \
    --region=$REGION \
    --timeout=3600

echo ""

# Update Worker Service  
echo "⏱️  Updating scout-agent-worker timeout..."
gcloud run services update scout-agent-worker \
    --project=$PROJECT_ID \
    --region=$REGION \
    --timeout=3600

echo ""
echo "✅ Done! Verify with:"
echo "gcloud run services describe scout-agent-worker --project=$PROJECT_ID --region=$REGION --format='value(spec.template.spec.timeoutSeconds)'"
