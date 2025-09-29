#!/bin/bash

# Run the main ScoutAgent workflow
# Run this in Terminal 2 after starting MCP servers

CONTAINER_NAME="scout-agent-interactive"

echo "🎯 Running ScoutAgent Workflow"
echo "=============================="

# Check if container is running
if ! sudo docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
    echo "❌ Container $CONTAINER_NAME is not running!"
    echo "Please run: ./run_interactive.sh first"
    exit 1
fi

# Default parameters (you can modify these)
TARGET_MARKET="${1:-Knowledge management tools}"
KEYWORDS="${2:-bidirectional links,markdown sync}"
SUBREDDITS="${3:-PKMS}"
PER_QUERY_LIMIT="${4:-1}"

echo "Parameters:"
echo "  Target Market: $TARGET_MARKET"
echo "  Keywords: $KEYWORDS"
echo "  Subreddits: $SUBREDDITS"
echo "  Per Query Limit: $PER_QUERY_LIMIT"
echo ""

echo "🚀 Starting ScoutAgent workflow..."
sudo docker exec $CONTAINER_NAME bash -c "if [ -f .env ]; then set -a; . ./.env; set +a; while IFS= read -r line; do if [[ \$line =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then key=\$(echo \"\$line\" | cut -d'=' -f1); value=\$(echo \"\$line\" | cut -d'=' -f2-); value=\$(echo \"\$value\" | sed \"s/^['\\\"]//; s/['\\\"]$//\"); export \"\$key=\$value\"; fi; done < .env; fi; python -m scout_agent.main --target-market '$TARGET_MARKET' --keywords '$KEYWORDS' --subreddits '$SUBREDDITS' --per-query-limit '$PER_QUERY_LIMIT'"
