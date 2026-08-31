# Technology overview

## Python

Python is the main implementation language. It provides a large scientific and
automation ecosystem while keeping controller and analysis logic readable.
Dataclasses, enums and type annotations represent observations, state and
commands. The same modules can be exercised by live adapters, synthetic unit
tests and saved-run replay.

## OpenCV and NumPy

OpenCV supplies image processing: color-space conversion, thresholding,
connected components, template matching, geometry and visual diagnostics.
NumPy represents captured frames as arrays and makes pixel-level operations
fast enough for a responsive perception loop.

## MSS and Win32 APIs

MSS captures the selected window rectangle efficiently. pywin32 and `ctypes`
provide HWND discovery, foreground validation, cursor inspection and explicit
keyboard/mouse adapters. Window identity and coordinate transforms are checked
before input so the controller fails closed instead of falling back to an
arbitrary desktop target.

## OCR and visible telemetry

The project combines a fast template-based reader for stable UI glyphs with
Tesseract for less constrained text. A small Lua addon renders already-visible
client events into fixed color or text strips. The external controller still
learns them only from captured pixels, which preserves the screen-only runtime
boundary while making noisy UI signals testable.

## Ultralytics YOLO and PyTorch

YOLO provides optional object proposals for visible ore and other screen
objects. PyTorch supplies model inference and CUDA acceleration. The model is a
candidate generator rather than an action authority: deterministic tooltip,
cursor and postcondition checks remain responsible for clicks. Training tools
support reviewed labels, hard negatives and episode-level splits.

## Route geometry and navmesh planning

The runtime uses cyclic polyline projection, monotonic unwrapped progress,
lookahead steering and cross-track measurements. Offline tools can consume
Detour-compatible navmesh and terrain references to build route entry paths and
hazard-aware cycles. Those heavy third-party inputs are deliberately external
to this repository.

## Immutable evidence, supervisor and controllers

Temporal state machines convert incomplete visual evidence into bounded
behavior. V0.9 builds one immutable `WorldSnapshot`, then a pure priority
supervisor selects exactly one domain controller per tick. Controllers return
explicit command schedules; they do not capture frames, sleep or touch the OS.
This is conceptually similar to robotics behavior arbitration and industrial
RPA, where partial observability and safe recovery matter more than one perfect
frame classification.

## Serialized input execution

One nonblocking `InputExecutor` owns all side effects. It reconciles held
keys/buttons, enforces action windows and minimum gaps, and cancels stale
command generations when a higher-priority domain preempts the current one.
The live adapter pumps that one schedule between slower perception frames;
shadow and replay modes never create an OS input backend.

## Pytest and replay testing

Pytest covers pure geometry, immutable evidence, supervisor priorities,
controller transitions, executor scheduling, deterministic replay invariants
and recovered live regressions.
Heavy external models are replaced by deterministic fakes for controller tests.
Saved media is omitted from this snapshot, so external replay tests skip when a
licensed fixture is not available.

## YAML, JSONL and analysis tooling

YAML keeps behavior thresholds and feature gates outside source code. JSON and
JSONL store generated routes and explainable per-tick telemetry. Dedicated
scripts analyze routes, build datasets, render overlays and compare revisions,
making experimental conclusions reproducible rather than anecdotal.

## Packaging and CI

Setuptools provides the installable `src`-layout package. PyInstaller can build
a Windows executable for a controlled local setup. GitHub Actions installs the
project on a clean Windows runner, compiles the source and executes the offline
test suite on every push and pull request.
