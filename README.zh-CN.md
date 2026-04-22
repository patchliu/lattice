<p align="center">
  <img src="docs/assets/lattice-logo.svg" alt="Lattice logo" width="160" />
</p>

<h1 align="center">Lattice</h1>

<p align="center">
  面向大语言模型的 CPU-first、Rust-native 推理基础设施。
</p>

<p align="center">
  <a href="README.md">English</a>
  ·
  <a href="docs/architecture.md">架构文档</a>
  ·
  <a href="docs/roadmap.md">路线图</a>
  ·
  <a href="CONTRIBUTING.md">贡献指南</a>
</p>

## 项目概述

Lattice 是一个面向 CPU 推理场景的 Rust 系统工程项目。它关注模型加载、内存管理、Tensor 表达、Kernel 分发和运行时编排之间的边界清晰度，使推理链路在扩展过程中仍然具备可观察性、可维护性和可优化性。

项目目标是为本地部署和多节点部署提供一套面向工程落地的 LLM 推理基础。重点不在于拼接一个一次性的原型，而在于建立稳定的运行时骨架，为后续的算子优化、调度能力和分布式扩展提供可靠基础。

## 当前状态

当前仓库已经具备基础 workspace 结构、crate 拆分、CLI 入口、本地 `GGUF` 与 `Safetensors` 格式识别、基于 `mmap` 的权重映射，以及一个用于验证运行时骨架的占位 first-token 流程。

这意味着现在已经可以完整验证一条最小链路：打开本地模型文件、初始化 runtime、按需加载外部 kernel 动态库，并确认从 CLI 到 runtime 边界的调用流程是连通的。

## 仓库结构

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

## 快速开始

```bash
cargo fmt --all
cargo clippy --workspace --all-targets
cargo test --workspace
cargo run -p lattice-cli -- doctor
```

如果要验证当前运行时链路，可以使用本地模型文件执行：

```bash
cargo run -p lattice-cli -- run --model /path/to/model.gguf --prompt "hello lattice"
```

如果启动时还需要加载外部 kernel 动态库：

```bash
cargo run -p lattice-cli -- run \
  --model /path/to/model.gguf \
  --prompt "hello lattice" \
  --kernels /path/to/libkernels.so
```

## 文档

- [README.md](README.md): 英文版首页
- [docs/architecture.md](docs/architecture.md): 模块边界与运行时结构
- [docs/roadmap.md](docs/roadmap.md): 里程碑与演进路径
- [docs/detailed-plan.md](docs/detailed-plan.md): 详细实施规划
- [CONTRIBUTING.md](CONTRIBUTING.md): 协作方式
