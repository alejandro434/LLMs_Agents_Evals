# LangGraph Server Configuration

This directory contains the configuration files needed to launch a LangGraph server
for the BuildWithin Evals project.

## Setup

1. **Environment Variables**: Copy the example environment file and configure it:
   ```bash
   cp langgraph_server/env.example .env
   ```

   Then edit `.env` in the root directory and add your API keys:
   - `LANGCHAIN_API_KEY`: Your LangChain API key
   - `OPENAI_API_KEY`: Your OpenAI API key (if using OpenAI models)
   - Other optional keys as needed

2. **Install Dependencies**: The project dependencies including `langgraph-cli` are
   defined in the root `pyproject.toml`. Install them using uv:
   ```bash
   uv sync
   ```

## Running the LangGraph Server

### Option 1: Using uv run with LangGraph CLI (Recommended)

From the **root of the repository**, choose one config below to launch a specific
graph:

```bash
# Router (default). Equivalent to langgraph_server/langgraph.json
uv run langgraph dev --config langgraph_server/router.json

# Planner/Executor subgraph
uv run langgraph dev --config langgraph_server/planner_executor.json

# RAG subgraph
uv run langgraph dev --config langgraph_server/rag.json

# ReAct subgraph
uv run langgraph dev --config langgraph_server/react.json

# Reasoner subgraph
uv run langgraph dev --config langgraph_server/reasoner.json

# Receptionist subgraph
uv run langgraph dev --config langgraph_server/receptionist.json
```

The original `langgraph_server/langgraph.json` also launches the router subgraph.

### Option 2: Using the Launch Script

Run the provided launch script from the root:

```bash
./langgraph_server/launch.sh
```

This script will:
- Check you're in the root directory
- Ensure dependencies are synced
- Create .env from template if needed
- Launch the server using uv run with the router config

### Option 3: Using LangGraph Studio

1. Open LangGraph Studio
2. Navigate to the root of the repository
3. Select any config under `langgraph_server/` (e.g., `router.json`, `rag.json`)
4. The studio will automatically load the corresponding subgraph

### Programmatic smoke test using RemoteGraph

Use the helper script to call a running subgraph with a schema-valid payload:

```bash
uv run AWARE/agent-evals/subgraph_evals.py \
  --graph rag_graph
  # For receptionist, use:
  # --graph receptionist
```

Notes:
- Start a subgraph server first, e.g.:
  - `uv run langgraph dev --config langgraph_server/rag.json`
  - Or pick any other config in `langgraph_server/` and set `--graph` to its id
- Docs: [RemoteGraph reference](https://langchain-ai.github.io/langgraph/reference/remote_graph/),
  [Use a remote graph](https://docs.langchain.com/langgraph-platform/use-remote-graph),
  [JS how-to](https://langchain-ai.github.io/langgraphjs/how-tos/use-remote-graph/)

## Graph Architecture

By default, the router subgraph is exposed from
`src/graphs/router_subgraph/lgraph_builder.py`.

Additional configs are provided to run each compiled subgraph independently:

- `langgraph_server/planner_executor.json` →
  `src/graphs/planner_executor_subgraph/lgraph_builder.py:subgraph`
- `langgraph_server/rag.json` →
  `src/graphs/rag_subgraph/lgraph_builder.py:subgraph`
- `langgraph_server/react.json` →
  `src/graphs/ReAct_subgraph/lgraph_builder.py:subgraph`
- `langgraph_server/reasoner.json` →
  `src/graphs/reasoner_subgraph/lgraph_builder.py:subgraph`
- `langgraph_server/receptionist.json` →
  `src/graphs/receptionist_subgraph/lgraph_builder.py:subgraph`

The router subgraph includes:

- **Router Node**: Routes user inputs to appropriate subgraphs
- **Planner Executor Subgraph**: Handles planning and execution tasks
- State management using `RouterSubgraphState`

## API Endpoints

Once the server is running, you can access:

- **GraphQL Playground**: `http://localhost:8000/graphql`
- **REST API**: `http://localhost:8000/docs`
- **WebSocket**: `ws://localhost:8000/ws`

## Testing the Server

You can test the server using curl:

```bash
curl -X POST http://localhost:8000/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "user_input": "Call rag, then call react and finally call the reasoner."
    }
  }'
```

For the receptionist subgraph (launched via `receptionist.json`), include a
`thread_id` to enable checkpointed persistence:

```bash
curl -X POST http://localhost:8000/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "messages": ["Hi, I\'m looking for an entry-level retail job."]
    },
    "config": {
      "configurable": { "thread_id": "demo-thread-1" }
    }
  }'
```

The receptionist graph is compiled with a persistent SQLite checkpointer
(`checkpoints.sqlite` in the repo root), so using the same `thread_id` will
resume context across calls. See the docs for details:

- Checkpointing reference: `https://langchain-ai.github.io/langgraph/reference/checkpoints/`
- Persistence concepts: `https://langchain-ai.github.io/langgraph/concepts/persistence/`

## Development

To modify the graph:
1. Edit the source files in `src/graphs/router_subgraph/`
2. The changes will be reflected when you restart the server
3. Test your changes using the test script in the main graph builder

## Troubleshooting

- **Port already in use**: Change the `LANGSERVE_PORT` in your `.env` file
- **Import errors**: Ensure you're running from the repository root with `uv run`
- **API key errors**: Verify your `.env` file has the correct API keys
- **langgraph command not found**: Run `uv sync` to install dependencies
- **Module not found**: Make sure you're using `uv run` prefix for all commands
