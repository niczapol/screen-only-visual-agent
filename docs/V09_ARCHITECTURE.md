# V0.9 Deterministic Runtime Architecture

Updated: 2026-08-28

## Decision

V0.9 is a replacement runtime, not another patch to `live_route_probe.py`.
New behavior is implemented through one screen-only, replayable kernel. The
v0.8 runner remains available only as an explicit compatibility/evidence path.

The project principles remain unchanged: autonomous progress, reproducible
decisions, smooth continuous movement, bounded recovery, low latency and no
hidden game state.

## Runtime Boundary

Allowed runtime inputs are the selected game-window pixels, visible addon
markers, the bot's own action/history state and keyboard/mouse output. Offline
maps, navmesh and spawn databases are compile-time references. Memory reads,
injection, packet inspection and hidden server APIs remain forbidden.

## Data And Control Flow

```text
ScreenCapture
  -> SnapshotBuilder
       -> WorldSnapshot(Evidence[T])
       -> protocol freshness / checksum / frame sequence
  -> physical hazard fusion
  -> phase-gated recovery and mining sensors
  -> Supervisor.reduce(memory, snapshot)
       -> exactly one CommandGroup owner
  -> selected pure domain controller
       -> ControlIntent(Command...)
  -> InputExecutor
       -> generation cancellation / held-state reconciliation
  -> Windows input backend (live only)
  -> replay-compatible JSONL event
```

`WorldSnapshot` is immutable. Every evidence item records state, confidence,
time, frame and source. `UNKNOWN` is not converted to `ABSENT`. Window/input
safety, life and each domain observation retain separate authority. Stale
protocol withdraws pose and therefore route/mining control, but does not
falsify a safe selected input adapter or suppress independent visible combat/
recovery evidence.

The input executor is the only component allowed to apply commands. Controller
code contains no sleeps and cannot capture a frame or press a key. Preemption
cancels older command generations and releases incompatible held input. Shadow
mode never constructs the executor.

Live mode runs that same executor behind one thread-safe 12 ms actuator pump.
The pump is not a second decision maker: it only applies the current serialized
command schedule between slower capture/OCR frames. RMB yaw is a latest-
observation lane, so a new drag cancels the obsolete pending drag and releases
RMB without dropping compatible forward `W`. This prevents both queued-yaw
overshoot and the opposite failure where every new observation cancels a drag
before its scheduled mouse moves execute.

Snapshots retain capture time for evidence freshness. When the kernel submits
an intent after perception, the executor rebases relative `not_before` and
deadline windows to actual actuation time while retaining their internal
ordering. This prevents OCR latency from expiring an otherwise valid dialog
click and preserves semantic timing such as the released-key F-to-V gap.
Runtime logs record actuation delay, pending/held state and each executed or
expired event.

When a bounded live frame budget ends, the CLI does not immediately abandon
the last route position. It revokes pending input and continues observing with
a restricted executor scope: only combat, loot and recovery may act; travel,
mount and mining are prohibited. Five continuously quiet seconds are required
before normal teardown, with a separately bounded 600-second emergency drain.
This is a shutdown safety policy, not a new controller or input authority.

## Ownership Order

The supervisor selects one owner per frame:

1. death/recovery or a recognized recovery modal;
2. confirmed-combat emergency HP handling;
3. confirmed pending corpse loot;
4. confirmed combat, except the bounded ordinary mounted-escape case;
5. physical/visible forbidden-zone egress;
6. admitted mining transaction;
7. mount/remount;
8. normal route travel.

Ordinary mounted aggro with fewer than two visible attackers may retain travel
for at most 12 seconds. Critical HP, two-plus attackers, dismount or timeout
hands off to combat. Combat keys remain serialized `F -> V > X > Q > E` and
cannot overlap.

## Physical Route Contract

Percent-map coordinates are converted through an explicit `ZoneGeometry`
(`6900 x 4600` yards for Tanaris). Route projection, lookahead, mining radii,
hazard margins and access audits use yards rather than mixed percent/yard
constants.

The compiler requires:

- a matching route schema and zone;
- at least two usable points;
- no route point or closing segment inside any configured physical hazard;
- no admitted mining node or access option inside/crossing a hazard;
- permanent exclusions applied before controller construction;
- stored source SHA and route status for preflight.

V18 fails this contract because 25 waypoints cross older configured hazards.
V19 removes the disconnected unsafe excursion and inserts a validated bridge.
It is an offline candidate, not a production route, until live acceptance.

## Mining Without Microsteps

The old final approach mixed a DB anchor, stopped coordinate refreshes, one
calculated burst and a frozen marker residual. It avoided indefinite tapping,
but still created extra transitions and could steer from stale residual data.

V0.9 treats the two coordinate sources according to what each actually proves:

- the offline DB anchor identifies the expected spawn and safe access plan;
- the current minimap vector estimates the vein's live relative bearing;
- neither source proves the exact world mesh/click point;
- the native tooltip/pickaxe remains interaction authority.

Admission requires two stable native-marker frames, native tooltip/type, a
same-type V19 node, a safe access plan and proximity to its attachment. The
controller follows that access path while holding `W`. Each fresh tracked
marker updates the fused target. RMB yaw is proportional and bounded; `W` is
released only for a pivot, stale/unknown evidence, a centered marker or the
4.5-yard capture envelope.

Inside the capture envelope, the bot stops once and searches the visible world.
The detector proposes hover points asynchronously. A click occurs only at a
point where the native cursor/tooltip proves mining, the cursor is within 48
pixels of the current model proposal and both are inside the configured world-
interaction region. Minimap and action-bar coordinates are excluded. The
executor allows 0.12 seconds for a moved cursor to settle before clicking.
Success requires loot or stable marker/hover clearance; combat aborts the
transaction and resets that clearance streak. Exactly one short W tap is legal
only after an explicit visible `out_of_range` outcome; it is recovery from
measured game feedback, not normal navigation.

This cannot be replaced safely by a blind DB-coordinate fly-in: the database
does not encode the exact current vein mesh, height, collision side or player-
icon occlusion. It also should not be minimap-only: the marker jitters and is
occluded at center. Fusion keeps the stable identity/plan and the fresh local
bearing without turning either into click authority.

## Perception And Addon Protocol

Addon v0.5.1 renders protocol v2 as 64 payload bits plus an 8-bit checksum:
version, rolling frame sequence, zonal X/Y, heading, event sequence and status
flags. Runtime observes only captured pixels. The rolling frame sequence makes
a frozen but visually valid overlay distinguishable from a current one; the
event sequence makes combat/loot outcomes edge-aware.

World-ore inference has a single worker and no stale backlog. A search
generation change invalidates an older future result. Model boxes are proposals
only and never bypass native hover authority.

The telemetry ROI covers the client strip after UI scaling rather than assuming
the addon's logical width is its physical captured width. A saved scaled-strip
regression protects the sentinel/checksum boundary. V0.9.1 passive shadow
decoded all 100 final frames as fresh protocol v2.

This client can temporarily return no map coordinates during combat. The addon
retains its last valid coordinate/heading only while `UnitAffectingCombat` is
visibly encoded as true; outside combat it still withholds pose. This narrow
cache and the independent input-ready evidence keep combat operable without
turning stale coordinates into navigation authority.

On ordinary alive frames the expensive ghost-palette path is skipped. Legacy
damage/facing/nameplate scans are not invoked when their configured authorities
are disabled. Idle ore recognition runs every second 10 Hz frame, but any
active mining phase restores every-frame recognition. Measured mean end-to-end
frame processing fell from about 109 ms to 28 ms.

## Recovery Contract

A recognized dead/ghost live start is permitted only for exclusive recovery
ownership; ordinary route/mount/mining startup gates do not run in that state.
Recovery phases are monotonic, so generic healer-dialog evidence cannot regress
Return to Life or Accept and a terminal failure cannot restart itself.

`C` and the visible selected-healer addon marker establish target identity.
Only then may the controller use the nearest configured cemetery anchor as a
screen-coordinate-independent bearing reference, turn from visible heading and
advance in bounded `W` segments. `G` is emitted only inside the configured
six-yard radius. The anchor is a navigation reference, not hidden live state;
the visible selected target remains authority. F6 follow and blind forward
approach are rejected by live evidence.

After Return to Life/Accept and visible alive confirmation during an active
live recovery, the orchestrator persists a 600-second Resurrection Sickness
deadline. Until expiry it reports `resurrection_sickness`, gives no controller
input ownership and survives process restart. Safe-zone resurrection remains
disabled.

## Modes And Gates

- `preflight`: read-only static validation and short-lived manifest.
- `install-addon`: copies/verifies the addon without building the runtime.
- `replay`: executes recorded snapshots through the production kernel.
- `shadow`: captures and logs through the kernel without input capability.
- `live`: requires `--enable-live-input` and a fresh matching manifest.
- `legacy-gui`: isolated v0.8 compatibility path with its own explicit enable.

The preflight fingerprint covers the runtime source tree/package metadata, full
config and exact route, hazards, permanent exclusions, model, addon source and
installed addon. Validation may use an offline-candidate route; production requires route status
`live_validated`, `accepted` or `production`.

Live startup additionally requires current protocol v2 and foreground HWND
confirmation. Ordinary operation requires visible alive state, clear modal/
combat/forbidden state, usable durability, correct zone/position and a start
inside the configured 18-yard route corridor. Recognized dead/ghost state
bypasses only those ordinary gates and enters recovery exclusively. The runtime
CLI never starts the game client. This public snapshot omits machine-specific
launch configuration and keeps live input behind an additional disabled
portfolio gate.

## Rejected Alternatives

- Continue patching the v0.8 monolith: rejected because ownership, perception
  and side effects remain too coupled for deterministic replay.
- Use only the DB coordinate and interact blindly: rejected because it is a
  spawn reference, not a guaranteed reachable 3D interaction point.
- Chase only the minimap dot: rejected because center occlusion and projection
  jitter can create oscillation.
- Let the world model click: rejected because proposal confidence is not native
  interaction evidence.
- Replace all logic with one learned end-to-end policy: deferred because the
  current dataset cannot prove rare safety transitions or reproduce decisions.
- Preserve all V18 coverage in one loop: rejected for v0.9 because the global
  hazard audit disproves its safety. Restore coverage as regional loops.

## Acceptance Sequence

1. Unit/saved-frame tests and deterministic replay.
2. Full repository suite and fresh static validation preflight.
3. Passive shadow against the reinstalled client. **Passed in v0.9.1.**
4. Short supervised V19 traversal from a repaired safe rail start. V0.9.2 has
   partial smooth route/approach evidence, but its combat interruption prevents
   route acceptance; a post-fix run is still required.
5. Several continuous mining transactions with measured stops/turns/outcomes.
6. Mounted threat/combat/loot validation.
7. Natural death, recovery, sickness wait and same-route resume. Recovery from
   dead/ghost through Return/Accept passed in v0.9.2 after cursor-settle repair;
   same-route resume remains pending.
8. Longer multi-node and unattended runs only after the preceding gates pass.
