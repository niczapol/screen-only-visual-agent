# Engineering evidence

This snapshot preserves the code and lightweight route artifacts while
excluding raw operational media. The following measurements come from the
source project's retained test and run reports.

## Verified implementation evidence

- Final archived source-project suite: 898 passed, 22 skipped.
- Sanitized v0.9.2 portfolio suite: 885 passed, 35 explicitly skipped because the
  corresponding private media or external navigation inputs are omitted.
- Static source inventory at export: 97 runtime Python modules, 55 Python tools
  and 103 test modules.
- V19 route artifact: 304 stored route entries, 303 normalized physical points,
  74 planned nodes and 73 effective candidates after permanent exclusions.
- Exact all-hazard audit: zero V19 route points, closing segments, nodes or
  access options inside configured hazards. V19 remains an offline candidate,
  not a production-acceptance claim.
- Best long directional window: 47.206% projected coverage with zero accepted
  progress regression or off-corridor observations.
- One fully verified autonomous interaction transaction: one authorized click,
  visible completion/clearance, cooldown and route resume.
- Multiple complete visible-dialog recovery transactions through target,
  interaction, confirmation and persisted cooldown states.
- A bounded database match benchmark remained below one millisecond in recorded
  development telemetry.

## Representative problems solved

- Backward route oscillation caused by projecting onto a nearby parallel loop
  segment.
- Lap-seam progress jumps caused by geometry that was spatially close but
  logically almost one complete lap ahead.
- Mouse-facing sign inversion between world-coordinate steering and direct
  screen-space targeting.
- False state transitions caused by one missing frame, stale target UI and
  quantized health-bar measurements.
- Conflicting combat, mining, loot and recovery actions in the same controller
  tick.
- Repeated interaction against the same failed physical target after coordinate
  projection jitter.
- Runtime model dependency drift between preflight and the direct runner.
- Stale queued mouse drags and a control-loop starvation failure, replaced by a
  latest-observation yaw lane and a single 12 ms actuator pump.
- Valid commands expiring during perception, fixed by rebasing relative action
  windows at executor submission while preserving their internal order.
- Unbounded storage growth from screenshots and video, replaced with event
  buffers, manifests and reviewed compaction.

## Honest status

The project demonstrates an end-to-end architecture and individual successful
transactions, but it is not presented as a finished unattended product.
Repeatable mining, full-loop acceptance, multi-enemy combat and long-run repair
workflows remain development gates.
