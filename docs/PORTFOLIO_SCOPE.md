# Portfolio scope and data policy

This repository was exported from a larger local research workspace. The goal
is to make the engineering inspectable without publishing personal data,
copyrighted media or tens of gigabytes of generated artifacts.

## Included

- original Python controller, perception and analysis code;
- original Lua telemetry addon under a neutral project name;
- unit and replay-oriented test code;
- generated JSON/CSV route outputs required by tests;
- a synthetic spawn database and a synthetic offline replay;
- safe example configuration, CI and documentation.

## Excluded

- user/account names, local profile paths and launcher shortcuts;
- screenshots, videos, logs and live-run manifests;
- downloaded web images and client-derived cursor artwork;
- game archives, extracted maps, DBC/ADT files and server SQL;
- external navmesh data and source repositories;
- trained model weights and training datasets;
- private operational history and internal handoff documents.

## Reproduction boundary

The committed offline route example and tests are reproducible from this
repository. The complete local live research environment is intentionally not
reproducible from the portfolio snapshot alone because its external assets may
have separate licenses and may contain private operational data.

