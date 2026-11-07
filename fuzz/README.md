# ACPI AML and resource fuzz targets

These host-side targets exercise AML table parsing, bounded AML method execution,
token-relative native-region access, and resource-template decoding. Inputs are
bounded to the FirmwareRoot single-table limit, a 64 KiB method body, and a
smaller resource-template limit. Both AML targets install a bounded allocation
and execution meter. `aml_execute` also uses a native-region test backend and
can revoke each issued token before access.

Use the pinned NaOS Rust stage1 compiler and a local `cargo-fuzz` installation.
The custom compiler in this workspace has no Address Sanitizer runtime, so use
`--sanitizer none`; libFuzzer's coverage instrumentation remains enabled.
Set the two build-directory variables to the directories used for your Rust
toolchain and fuzz tooling, then run from the repository root:

```sh
NAOS_RUST_BUILD=build
ACPI_FUZZ_BUILD=build-acpi-pci
export CARGO="$PWD/$NAOS_RUST_BUILD/rust-naos/x86_64-unknown-linux-gnu/stage1-tools-bin/cargo"
export RUSTC="$PWD/$NAOS_RUST_BUILD/rust-naos/x86_64-unknown-linux-gnu/stage1/bin/rustc"
export RUSTC_BOOTSTRAP=1
export PATH="$PWD/$ACPI_FUZZ_BUILD/cargo-fuzz-tools/bin:$PATH"

mkdir -p "$ACPI_FUZZ_BUILD/aml-fuzz-corpus" \
    "$ACPI_FUZZ_BUILD/aml-execute-fuzz-corpus" \
    "$ACPI_FUZZ_BUILD/resource-fuzz-corpus" \
    "$ACPI_FUZZ_BUILD/aml-fuzz-artifacts" \
    "$ACPI_FUZZ_BUILD/aml-execute-fuzz-artifacts" \
    "$ACPI_FUZZ_BUILD/resource-fuzz-artifacts"
cargo-fuzz run --fuzz-dir naos/third_party/rust-acpi-naos/fuzz \
    --target-dir "$ACPI_FUZZ_BUILD/aml-fuzz-target" --sanitizer none \
    aml_load "$ACPI_FUZZ_BUILD/aml-fuzz-corpus" -- \
    -artifact_prefix="$ACPI_FUZZ_BUILD/aml-fuzz-artifacts/" \
    -max_total_time=60 -max_len=1048576 -timeout=5 -rss_limit_mb=512
cargo-fuzz run --fuzz-dir naos/third_party/rust-acpi-naos/fuzz \
    --target-dir "$ACPI_FUZZ_BUILD/aml-execute-fuzz-target" --sanitizer none \
    aml_execute "$ACPI_FUZZ_BUILD/aml-execute-fuzz-corpus" -- \
    -artifact_prefix="$ACPI_FUZZ_BUILD/aml-execute-fuzz-artifacts/" \
    -max_total_time=60 -max_len=65536 -timeout=5 -rss_limit_mb=512
cargo-fuzz run --fuzz-dir naos/third_party/rust-acpi-naos/fuzz \
    --target-dir "$ACPI_FUZZ_BUILD/resource-fuzz-target" --sanitizer none \
    resource_template "$ACPI_FUZZ_BUILD/resource-fuzz-corpus" -- \
    -artifact_prefix="$ACPI_FUZZ_BUILD/resource-fuzz-artifacts/" \
    -max_total_time=60 -max_len=65536 -timeout=5 -rss_limit_mb=512
```

The corpora, target artifacts, and crash artifacts should stay under the
selected build directory. Keep useful regression inputs in the build-local
corpora during a run; promote a reproducing input to a named deterministic test
before relying on it as a long-term regression.

The parent NaOS crate also runs deterministic single-bit mutation regression
tests through the budgeted executor and the resource parser. These fuzz targets
complement, rather than replace, those contract tests.
