# Shepherd Run Failure — 2026-06-22

**Timestamp (UTC):** 2026-06-22  
**Status:** FAILED — A800 unreachable

## Root cause

The shepherd container (claude.ai/code remote execution environment) does not have
`ssh` or `sshpass` binaries installed, and no SSH-capable MCP server is configured
for this session. Every attempt to reach `zeyuwang@117.74.66.181:50507` returned:

```
/bin/bash: line 1: sshpass: command not found
/bin/bash: line 1: ssh: command not found
```

## Impact

- state.json **not read** — queue/running status unknown.
- GPU utilization **not checked**.
- No stuck-experiment detection, no result analysis, no disk-pressure check.
- No actions taken (A, B, C, D, E all skipped).
- shepherd.log on A800 **not updated** (no write path available).

## Required fix

For shepherd to function, one of the following must be true in its execution environment:

1. `ssh`/`sshpass` binaries present in the container PATH, **or**
2. An SSH-capable MCP server (e.g. `mcp-server-ssh`) configured in the session, **or**
3. A `claude-code-remote` MCP server with a `remote_exec` tool wired to A800.

Please update the shepherd session configuration to include SSH access.
