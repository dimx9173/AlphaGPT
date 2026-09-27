Act as Claude architect for a PRPCC cycle in /home/brian/project/AlphaGPT. Read .prpcc/regime-gate/PRP.md and .prpcc/regime-gate/PLAN.md. Do NOT edit files and do NOT run long jobs.

Context: the 28c GA winner is a ~91% short strategy. The current gate min_oos_portfolio_sharpe=1.0 is absolute, and on the lockbox the equal-weight market itself scored Sharpe 1.120 during a +55% rally, so the gate was measuring regime rather than skill. The PRP replaces the blocking Sharpe bar with walk-forward folds inside train+validation and a Sharpe-of-the-difference-series excess metric.

Return: (1) feasibility of F1-F6 and Task 1-5; (2) exact implementation order and the function signatures you recommend; (3) mathematical correctness of the excess-return approach and the regime label; (4) hidden edge cases, especially fold-boundary artifacts, float-sign flapping at zero benchmark return, and the embargo gap; (5) test cases I should insist on; (6) any place where the spec could be gamed to make the v3c formula pass.

PRP.md section 2 is a binding anti-gaming clause: thresholds are frozen and the gate is allowed to fail v3c. Critique the gate design, do not help me tune it to pass.