#!/bin/bash

# Test script to verify environment variables are loaded in the container

CONTAINER_NAME="scout-agent-interactive"

echo "🔍 Testing environment variables in container..."

sudo docker exec $CONTAINER_NAME bash -c "
if [ -f .env ]; then
    echo '📋 Loading .env file...'
    set -a
    . ./.env
    set +a
    echo '✅ Environment variables loaded'
else
    echo '❌ No .env file found'
fi

echo '🔍 Checking API keys:'
echo 'SCOUT_OPENAI_API_KEY:' \${SCOUT_OPENAI_API_KEY:+SET} \${SCOUT_OPENAI_API_KEY:-NOT SET}
echo 'SCOUT_ANTHROPIC_API_KEY:' \${SCOUT_ANTHROPIC_API_KEY:+SET} \${SCOUT_ANTHROPIC_API_KEY:-NOT SET}
echo 'SCOUT_GEMINI_API_KEY:' \${SCOUT_GEMINI_API_KEY:+SET} \${SCOUT_GEMINI_API_KEY:-NOT SET}
echo 'SCOUT_DEEPSEEK_API_KEY:' \${SCOUT_DEEPSEEK_API_KEY:+SET} \${SCOUT_DEEPSEEK_API_KEY:-NOT SET}
echo ''
echo '🔍 Checking Reddit credentials:'
echo 'SCOUT_REDDIT_CLIENT_ID:' \${SCOUT_REDDIT_CLIENT_ID:+SET} \${SCOUT_REDDIT_CLIENT_ID:-NOT SET}
echo 'SCOUT_REDDIT_CLIENT_SECRET:' \${SCOUT_REDDIT_CLIENT_SECRET:+SET} \${SCOUT_REDDIT_CLIENT_SECRET:-NOT SET}
echo 'SCOUT_REDDIT_USER_AGENT:' \${SCOUT_REDDIT_USER_AGENT:+SET} \${SCOUT_REDDIT_USER_AGENT:-NOT SET}
echo 'SCOUT_REDDIT_USERNAME:' \${SCOUT_REDDIT_USERNAME:+SET} \${SCOUT_REDDIT_USERNAME:-NOT SET}
"
