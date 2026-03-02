---
name: tmux
description: "Remote-control tmux sessions for background tasks and persistent processes"
requires:
  bins: ["tmux"]
---

# tmux Session Control

## Creating Sessions

```bash
# Named session with a command
tmux new-session -d -s mywork "long-running-command"

# Plain session
tmux new-session -d -s scratch
```

## Reading Output

```bash
# Capture visible pane content
tmux capture-pane -t mywork -p

# Capture with scrollback (last 1000 lines)
tmux capture-pane -t mywork -p -S -1000
```

## Sending Input

```bash
# Send keystrokes
tmux send-keys -t mywork 'ls -la' Enter

# Send Ctrl+C to interrupt
tmux send-keys -t mywork C-c
```

## Session Management

```bash
# List sessions
tmux list-sessions

# Kill a session
tmux kill-session -t mywork

# Check if a session exists
tmux has-session -t mywork 2>/dev/null && echo "exists" || echo "gone"
```

## Window & Pane Control

```bash
# Split pane horizontally
tmux split-window -h -t mywork

# Split pane vertically
tmux split-window -v -t mywork

# Select a specific pane
tmux select-pane -t mywork.1

# Create a new window in a session
tmux new-window -t mywork -n logs
```

## Patterns

- Use tmux for any command that might take more than a few seconds — it lets you check progress without blocking.
- Name sessions descriptively so you can manage multiple concurrent tasks.
- Always check `tmux list-sessions` before creating new ones to avoid duplicates.
- Use `tmux capture-pane` to read output rather than running commands synchronously when you want non-blocking status checks.
