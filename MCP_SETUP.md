# MCP Setup

This project includes project-scoped Codex MCP servers in `.codex/config.toml`.
Codex loads project `.codex/config.toml` only when the project is trusted.

## Servers

- `github`: official GitHub MCP server via Docker image `ghcr.io/github/github-mcp-server`.
- `python_sandbox`: Python sandbox MCP server via `uvx mcp-run-python==0.0.22 stdio`.
- `filesystem`: local filesystem MCP server scoped to this project folder only.

## Prerequisites

- Docker running for the GitHub MCP server.
- A GitHub personal access token in your environment as `GITHUB_PERSONAL_ACCESS_TOKEN`.
- `uvx` and Deno available for `mcp-run-python`.
- Node/npm available for the filesystem server's `npx` command.

## Token Setup

Add this to your local shell profile or uncommitted `.env` file:

```bash
GITHUB_PERSONAL_ACCESS_TOKEN=your_token_here
```

Do not commit real tokens. `.env` is already ignored by `.gitignore`.

## Check In Codex

After restarting Codex in this trusted project, run:

```text
/mcp
```

You should see `github`, `python_sandbox`, and `filesystem` listed.
