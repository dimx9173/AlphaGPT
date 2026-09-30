# Known pre-existing test failures (not caused by the factor work)

Observed 2026-09-30 while adding the causality guard.

## Broker and venue tests fail because pytest-asyncio is not installed

    ModuleNotFoundError: No module named 'pytest_asyncio'

    "async def functions are not natively supported"

Affected: `tests/test_cex_brokers.py`, `tests/test_aster_broker.py`,
`tests/test_hyperliquid_broker.py`, `tests/test_venue_base.py`,
`tests/test_jupiter.py`.

Every one of these is an `async def` test that pytest cannot collect without
an async plugin. They are not testing anything and they are not passing. This
was confirmed by stashing all work from the factor task and re-running them:
they fail identically on the previous commit.

The fix is one dependency:

    uv pip install pytest-asyncio

and a `asyncio_mode = auto` (or per-test markers) in the pytest config. It has
been left undone deliberately: it is outside the scope of the factor work, and
silently installing a test dependency would change what CI means for every
other test in the repo without anyone deciding that.

## These tests cannot reach a broker in this environment regardless

The broker tests also require credentials and a live or testnet endpoint. Even
with the plugin installed they would fail closed here, which is the correct
behaviour for this project. The suite that covers the 28c research path
(`-k "28c or causality or accounting or splits or holdout or factor or ga_"`)
is 260 tests and passes clean.

## Three other failures are real and unrelated

    tests/test_e10_gates.py::test_paper_baseline_within_tolerance
    tests/test_iter_h2.py::test_iter_h2_weights_arms_present
    tests/test_iter_h2.py::test_iter_h2_weights_script_runs_offline
    tests/test_iter_y1.py::test_iter_y1_weights_arms_present
    tests/test_iter_y1.py::test_iter_y1_weights_script_runs_offline

These are ordinary assertion failures, not collection errors, and they also
fail on the previous commit. They concern a paper baseline tolerance and the
h2/y1 iteration weight arms. They are recorded here so a future session does
not re-derive that they predate the factor work.
