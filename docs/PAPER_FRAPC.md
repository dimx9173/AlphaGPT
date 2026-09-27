# AlphaGPT 28-coin paper deployment

- Target: `brian@178.105.232.221`
- Remote root: `/home/brian/project/AlphaGPT`
- Mode: broker-free paper replay only
- Universe: 28 local 30m symbols, timestamp-aligned intersection (17,519 bars)
- Formula: frozen transfer candidate from the five-coin training artifact
- Remote schedule: `5 * * * *` via `/home/brian/project/AlphaGPT/research/run_paper_remote.sh`
- Latest output: `results/paper_latest.json`
- No `.env`, API keys, Y1B state, broker modules, or order commands are used.

The local AlphaGPT Y1B hourly cron was removed after backing up the crontab. Existing local demo state was not closed or deleted.

This is a transfer diagnostic, not a newly refit 28-coin production strategy. The formula was trained on ETC/TRX/ATOM/APT/KAS; all 28 names use explicit common transfer thresholds. The runner uses causal features, a one-bar execution lag, frozen-formula sequential OOS segments, and a capital-weight cap.

Rollback: remove the remote paper cron line, then restore the local crontab backup if needed.
