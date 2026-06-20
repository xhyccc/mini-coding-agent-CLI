# codelet examples

Runnable examples that drive the **real** `codelet` agent against the
OpenAI-compatible endpoint configured in your repository-root `.env` file.

## Prerequisites

Create a `.env` at the repository root with your provider credentials:

```
LLM_PROVIDER=custom
LLM_BASE_URL=https://your-endpoint/v1
LLM_MODEL=your-model-name
LLM_API_KEY=sk-...
```

The API key is read straight from `.env` (or a real environment variable) and
is never printed by these scripts.

## Files

- `_runner.py` — shared helpers. `build_model_client_from_env()` constructs an
  `OpenAIModelClient` from `.env`; `build_agent_from_env(workdir, **kwargs)`
  returns a ready-to-run `MiniAgent` rooted at a scratch workspace.
- `quickstart_real_agent.py` — asks the agent to plan, read two files in
  parallel, and write a `summary.md`. Exercises the Iteration 1–5 tuning
  features (multi-tool batching, file-read dedup, no-progress breaker,
  argument repair, plan re-consultation).
- `run_gdpval_batch.py` — batch evaluation runner for the GDPval benchmark
  (220 tasks across Code / Shell / Data categories). Iterates through tasks,
  creates isolated workspaces, runs the codelet CLI agent, and collects
  deliverables. Progress is saved for resumable runs.

## Run

```bash
cd <repo-root>
python examples/quickstart_real_agent.py
```

The script creates a temporary workspace, runs the task, and prints the final
answer, stop reason, and the generated `summary.md`.

### GDPval Batch Evaluation

```bash
# Download the GDPval dataset first (one-time)
# See: https://github.com/swebench/GDPval

# Run all 220 tasks
python examples/run_gdpval_batch.py

# Run a subset
python examples/run_gdpval_batch.py --start 0 --end 50
```
