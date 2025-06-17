# Changelog

All notable changes to this project will be documented in this file.

## [0.1.1] - 2025-06-17

- Remove duplicate `submit_strangle` call in `run_strangle`.
- Add `RISK_PCT_PER_TRADE` to `apps/zero_dte/.env.example`.
- Update `zero_dte_app.py` documentation to reference dynamic sizing env var and updated `.env.example` path.

## [0.1.0] - 2025-06-15

- Add configurable iron-condor wing spread (`CONDOR_WING_SPREAD`).
- Clean up stray and missing return statements in zero_dte logic.
- Wire wing spread through Settings, schedule_for_symbol, and run_two_phase.
- Add .env.example and README for zero_dte app.
- Add unit tests for iron-condor helpers and end-to-end two-phase flows.

