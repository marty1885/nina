---
name: coding-agent
description: "Patterns for orchestrating AI coding agents like Claude Code, Codex, and similar tools"
requires:
  bins: ["claude"]
---

# Coding Agent Orchestration

## Spawning Coding Agents

When you need to write, modify, or analyze code, spawn a coding agent in a PTY session:

```bash
# Claude Code
tmux new-session -d -s code-task "claude --print 'implement the feature described in TASK.md'"

# Codex
tmux new-session -d -s code-task "codex 'implement the feature described in TASK.md'"
```

## PTY Patterns

- Always run coding agents inside `tmux` or `screen` so you can monitor progress and capture output.
- Use `tmux capture-pane -t <session> -p` to read the current output without interrupting.
- Use `tmux send-keys -t <session> 'input' Enter` to provide input if the agent prompts.

## Parallel Worktree Patterns

For independent tasks, use git worktrees to run multiple agents in parallel:

```bash
# Create isolated worktrees for parallel work
git worktree add ../feature-a -b feature-a
git worktree add ../feature-b -b feature-b

# Spawn agents in each
tmux new-session -d -s agent-a "cd ../feature-a && claude --print 'implement feature A'"
tmux new-session -d -s agent-b "cd ../feature-b && claude --print 'implement feature B'"

# Monitor both
tmux capture-pane -t agent-a -p
tmux capture-pane -t agent-b -p
```

## Progress Updates

- When running long coding tasks, periodically check agent output and relay progress to the user.
- Don't flood with updates — summarize at meaningful milestones (started, tests passing, done).
- If an agent appears stuck (no output for >60s), check on it and consider restarting with a refined prompt.

## Best Practices

- Give agents clear, scoped tasks. Vague prompts lead to vague results.
- Include relevant file paths and context in the prompt.
- Review agent output before committing — agents can introduce bugs or unnecessary changes.
- Prefer `--print` / non-interactive modes when you just need the result.
- Clean up worktrees after merging: `git worktree remove ../feature-a`.
