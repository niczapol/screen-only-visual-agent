# Security and safe operation

The committed configuration is intentionally fail-closed:

- live keyboard and mouse output is disabled;
- live route, mining, combat and recovery modes are disabled;
- machine-specific paths and credentials belong in ignored local files;
- model weights and external navigation data must be supplied separately.

The runtime is designed around visible pixels and explicit OS input. It does
not require process-memory reading, code injection, packet inspection or
hidden server telemetry.

Before enabling local live experiments, review the selected window, input
backend, key bindings, stop mechanism and all external data paths. Never run
the controller against software or services without authorization.

Please report security issues privately to the repository owner rather than
opening a public issue.

