# data-analyst-mcp

Reproducible data-analyst MCP server. Loads tabular data via DuckDB, runs proper EDA, hypothesis tests with assumption checking, and regression with diagnostics — and emits the full session as a runnable Jupyter notebook for audit and re-execution.

**Status**: under active development. See `docs/SPEC.md` for the implementation specification and `/home/oleg/.claude/plans/data-analyst-mcp-implementation-encapsulated-thacker.md` for the build plan.

## Not for

- Streaming data
- Dashboards / BI replacement
- Causal inference
- Datasets > 10 GB

## Install

```bash
uvx data-analyst-mcp
```

Then add to your `claude_desktop_config.json` — see `claude_desktop_config.example.json`.

## License

MIT.
