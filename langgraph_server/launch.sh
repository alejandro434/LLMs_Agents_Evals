#!/bin/bash
# Launch script for LangGraph server
# This script should be run from the root of the repository

# Check if we're in the root directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: This script must be run from the root of the repository"
    echo "Please cd to the root directory and try again"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found in root directory"
    echo "Creating .env from template..."
    cp langgraph_server/env.example .env
    echo "Please edit .env and add your API keys before running the server"
    exit 1
fi

# Sync dependencies to ensure langgraph-cli is installed
echo "Ensuring dependencies are up to date..."
uv sync

echo "Starting LangGraph server..."
echo "Configuration: langgraph_server/langgraph.json"
echo "Graph: Router Subgraph"
echo ""

# Launch the server using uv run
uv run langgraph dev --config langgraph_server/langgraph.json
