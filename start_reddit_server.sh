#!/bin/bash

# Start Reddit MCP server with proper environment loading
CONTAINER_NAME="scout-agent-interactive"

echo "🔧 Starting Reddit MCP Server with environment variables..."

sudo docker exec -d $CONTAINER_NAME bash -c "
# Load environment variables from .env file
if [ -f .env ]; then
    set -a
    . ./.env
    set +a
    # Export variables safely
    while IFS= read -r line; do
        if [[ \$line =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
            key=\$(echo \"\$line\" | cut -d'=' -f1)
            value=\$(echo \"\$line\" | cut -d'=' -f2-)
            value=\$(echo \"\$value\" | sed \"s/^['\\\"]//; s/['\\\"]$//\")
            export \"\$key=\$value\"
        fi
    done < .env
fi

# Start the Reddit MCP server
python -m scout_agent.mcp_integration.server.reddit_api
"

echo "✅ Reddit MCP server started with environment variables"
