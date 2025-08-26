### RemoteGraph subgraph smoke tests

Use `AWARE/agent-evals/subgraph_evals.py` to call any running subgraph with a
schema-valid payload.

#### 1) Start a subgraph server (example: RAG)

```bash
uv run langgraph dev --config langgraph_server/rag.json
```

The dev server listens at `http://127.0.0.1:2024` by default.

#### 2) Invoke via RemoteGraph

```bash
uv run AWARE/agent-evals/subgraph_evals.py --graph rag_graph \
  --url http://127.0.0.1:2024
```

Available graph ids: `router_graph`, `planner_executor_graph`, `rag_graph`,
`react_graph`, `reasoner_graph`.

#### RemoteGraph constructor (important)

Pass the graph id as the first positional argument (positional-only) and the
server URL as a keyword argument. Do not use `assistant_id=`.

```python
from langgraph.pregel.remote import RemoteGraph

graph = RemoteGraph("rag_graph", url="http://127.0.0.1:2024")
```

This matches the platform docs and allows the script to connect to the dev
server.

#### Valid inputs per subgraph

- `rag_graph`, `react_graph`, `reasoner_graph` expect a `current_step` with
  `Step` fields:

```json
{
  "current_step": {
    "instruction": "Search for information about Python and then analyze it",
    "suggested_subgraph": "rag",
    "reasoning": "First retrieve relevant documents, then extract the answer",
    "result": "",
    "is_complete": false
  }
}
```

- `planner_executor_graph` expects:

```json
{ "handoff_input": "Call rag, then react, then reasoner to find the answer." }
```

- `router_graph` expects:

```json
{ "user_input": "Call rag, then react, then reasoner to find the answer." }
```

#### Troubleshooting

- Error: `AttributeError: 'dict' object has no attribute 'instruction'`
  - Cause: server node code accessed `current_step` attributes without handling
    dict inputs from RemoteGraph.
  - Fix: coerce dicts to the `Step` model in server nodes. Example implemented
    in `src/graphs/rag_subgraph/nodes_logic.py`.

#### References

- RemoteGraph reference:
  `https://langchain-ai.github.io/langgraph/reference/remote_graph/`
- Use a remote graph:
  `https://docs.langchain.com/langgraph-platform/use-remote-graph`
- Using as a subgraph (same page):
  `https://docs.langchain.com/langgraph-platform/use-remote-graph`
