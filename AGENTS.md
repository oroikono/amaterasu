# AGENTS.md

## Project
This repo is the durable shared memory and coordination layer for the Amaterasu /
SymComp project.

Before doing substantive work, read:

- `handoff/CODEX_START.md`
- `handoff/CONTEXT.md`
- `handoff/PLAN.md`
- `handoff/DECISIONS.md`
- `handoff/TODO.md`

If the actual SymComp codebase is present, also read its `README.md`,
`EXPERIMENT_PLAN.md`, and test/cluster entry points before editing.

## Operating Rules
- Do not implement immediately after opening the repo. First summarize current
  understanding and identify missing context.
- Treat `handoff/TODO.md` and `handoff/DECISIONS.md` as durable memory. Update
  them when tasks or decisions change.
- Do not include secrets, tokens, passwords, SSH keys, API keys, or cluster
  credentials in the repo.
- Keep Euler-specific work compatible with durable storage and Slurm. Do not run
  heavy jobs on login nodes.
- Prefer small, reviewable changes.
- Before changing code, inspect the current implementation and run available
  checks when possible.

## Environments
- Alienware: local development, GPU smoke tests, and small training runs.
- MacBook Air: travel controller, editing, CPU tests, and review.
- ETH Euler: production cluster execution through Slurm and durable storage.
- Codex Web: planning, repo-level validation, and review.
- Codex VS Code/App: local and remote implementation.

## Euler Priority
The first implementation blocker is durable Euler storage. Outputs must not live
only under personal scratch. Use configurable environment variables for work and
home/archive paths rather than hardcoded private cluster paths.

## Handoff Discipline
At the end of meaningful work, update:

- `handoff/TODO.md` with next actions
- `handoff/DECISIONS.md` with durable decisions
- `handoff/PLAN.md` if the plan changed
