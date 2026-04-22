<p align="center">
  <img src="docs/assets/lattice-logo.svg" alt="Lattice logo" width="160" />
</p>

<h1 align="center">Lattice</h1>

<p align="center">
  CPU-first, Rust-native inference infrastructure for large language models.
</p>

<p align="center">
  <a href="README.zh-CN.md">简体中文</a>
  ·
  <a href="docs/architecture.md">Architecture</a>
  ·
  <a href="docs/roadmap.md">Roadmap</a>
  ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

## Overview

Lattice is an inference project for teams that want a clear, CPU-oriented systems stack in Rust. It is designed around explicit boundaries between model loading, memory management, tensor representation, kernel dispatch, and runtime orchestration so the execution path stays observable and extensible as the system evolves.

The project goal is to build a production-grade foundation for local and multi-node LLM inference on CPU infrastructure. That means prioritizing predictable data movement, practical runtime structure, and room for architecture-specific optimization instead of treating the runtime as a monolithic prototype.

## Current Status

The repository already provides the bootstrap workspace, crate layout, CLI entrypoints, local `GGUF` and `Safetensors` format detection, memory-mapped weight loading, and a placeholder bootstrap validation path for exercising the runtime skeleton.

This is enough to exercise the core integration surface end to end: open a local model file, initialize the runtime, optionally load an external kernel library, and verify the command flow from the CLI into the runtime boundary.

## Repository Layout

```text
.
├── crates/
│   ├── lattice-allocator
│   ├── lattice-cli
│   ├── lattice-core
│   ├── lattice-kernels
│   ├── lattice-model
│   ├── lattice-runtime
│   └── lattice-tensor
├── docs/
├── examples/
└── scripts/
```

## Quick Start

```bash
cargo fmt --all
cargo clippy --workspace --all-targets
cargo test --workspace
cargo run -p lattice-cli -- doctor
```

To exercise the current runtime path with a local model file:

```bash
cargo run -p lattice-cli -- run --model /path/to/model.gguf
```

If you have an external kernel library to load during startup:

```bash
cargo run -p lattice-cli -- run \
  --model /path/to/model.gguf \
  --kernels /path/to/libkernels.so
```

To exercise the model download + cache + mmap bootstrap flow:

```bash
cargo run --example load_model
```

## Documentation

- [README.zh-CN.md](README.zh-CN.md): Chinese overview
- [docs/architecture.md](docs/architecture.md): module boundaries and runtime structure
- [docs/roadmap.md](docs/roadmap.md): milestones and delivery sequence
- [docs/detailed-plan.md](docs/detailed-plan.md): implementation plan
- [CONTRIBUTING.md](CONTRIBUTING.md): contribution workflow
