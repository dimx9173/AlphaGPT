You are the Codex feasibility reviewer in a PRPCC cycle for AlphaGPT. Read .prpcc/regime-gate/PRP.md and .prpcc/regime-gate/PLAN.md in /home/brian/project/AlphaGPT. Do NOT modify files and do NOT run long jobs.

Audit the proposal against the current repository: research/accounting_28c.py, research/splits_28c.py, research/universe_28c.py, research/ga_28c_30m_3y.py, research/regime_gate_28c.py if present, and tests/.

Blocking things to look for: (a) is the excess-return Sharpe formulation mathematically sound and is any place implicitly subtracting Sharpes? (b) do walk_forward_folds as specified actually cover the train+validation region without touching the embargo or the lockbox? (c) does the design leak lockbox information into fold construction or into the gate? (d) is demoting min_oos_portfolio_sharpe from blocking to diagnostic defensible, or is it a disguised loosening? (e) is the verdict fail-closed? (f) any backwards-compatibility break in ACCEPTANCE_GATE_28C or in existing result JSON readers?

Return: VERDICT: PASS or CHANGES REQUIRED, then concise blocking findings with file:line references, then required clarifications.