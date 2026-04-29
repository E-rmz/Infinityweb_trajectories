"""Synthetic-task generator: turn MCTS trajectories into new tasks.

Pipeline:
  state_capture  →  diff  →  codegen  →  validator  →  builder

Each passing trajectory under ``thesis/traces/<env>/<task_id>/`` is turned
into one synthetic task at ``thesis/synthetic-tasks/<env>/<task_id>/path_NNN/``.
A synthetic task contains a generated ``verify(server_url)`` and matching
``solver(state)``, validated via golden-path + seed-fail + replay-pass.
"""
