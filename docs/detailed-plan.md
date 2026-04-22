# Lattice Detailed Plan

## 1. 项目定位

Lattice 的定位不是 “再写一个推理框架壳子”，而是构建一个以 Rust 为主、以 CPU 为优先目标、逐步向高并发和分布式扩展的推理运行时。仓库初始化阶段的重点是：

- 先定义工程骨架
- 先固定最小闭环
- 先让模块依赖方向稳定

## 2. Workspace 规划

当前 workspace 采用按职责拆 crate 的方式，而不是按阶段拆目录。这样做的原因是阶段会演进，但职责边界应尽量稳定。

### 当前 crate 边界

- `lattice-core`: 共享错误类型、请求 ID、通用 Result
- `lattice-allocator`: 对齐内存分配，后续承接 block 分配器
- `lattice-tensor`: Tensor shape / dtype / CPU buffer 抽象
- `lattice-model`: `GGUF` / `Safetensors` 权重来源与 `mmap` 映射
- `lattice-kernels`: 动态库加载与 kernel launcher 抽象
- `lattice-runtime`: L1 同步推理主流程与后续调度入口
- `lattice-cli`: 命令行 smoke test、开发调试入口

### 预计新增 crate

- `lattice-scheduler`: L2 的 continuous batching 与 block manager
- `lattice-tokenizer`: 异步 tokenizer 封装
- `lattice-quant`: L3 量化内核与格式适配
- `lattice-distributed`: L4 的 RPC、节点管理与状态迁移

## 3. L1 详细任务分解

### 3.1 Tensor & Memory

- 定义张量 shape、dtype 与线性存储布局
- 提供按 cache line / SIMD 要求对齐的内存分配
- 补齐 buffer 生命周期管理与安全边界说明

### 3.2 Kernel Launcher

- 定义动态库加载接口
- 定义 kernel 符号的 ABI 约定
- 准备最小可运行的 GEMM / Softmax 示例 kernel

### 3.3 Weight Loading

- 支持基于扩展名的格式检测
- 实现 `mmap` 零拷贝读取
- 后续补上 GGUF 元数据解析和 tensor 索引

### 3.4 Static Inference

- 构建同步 runtime builder
- 串起 model -> tensor -> kernel -> token 输出主流程
- 提供基础 tracing，保证每一步可观测

### 3.5 CLI / DX

- 提供 `doctor` 子命令验证工作区
- 提供 `run` 子命令走最小推理链路
- 为 benchmark、example、脚本留目录

## 4. L2 详细任务分解

### 4.1 Block Manager

- 设计固定大小 block 的元数据布局
- 支持 KV Cache 的分配、引用计数、回收
- 建立 block 与 tensor page 的映射关系

### 4.2 Continuous Batching

- 引入 Tokio 驱动的异步调度主循环
- 支持 request 在 decode 期间插入
- 将 tokenizer、prefill、decode 明确拆阶段

### 4.3 CPU PagedAttention

- 设计适合 CPU cache 的页大小
- 建立热点页重用策略
- 评估 L3 cache 共享行为对批量 decode 的影响

## 5. L3 详细任务分解

### 5.1 SIMD 与特化内核

- 先 AVX2，再 AVX-512 / AMX，再 ARM SVE
- 统一 kernel dispatch 接口，避免 runtime 分支爆炸

### 5.2 Quantization

- 优先支持 `GGUF K-Quants`
- 把解压与计算做成一体化 kernel，减少中间 buffer

### 5.3 NUMA

- 建立线程与内存近端绑定策略
- 在 runtime 中显式表达拓扑感知

## 6. L4 详细任务分解

### 6.1 RPC 与协议

这里需要先澄清一个技术决策问题：`Tonic` 基于 `gRPC/Protobuf`，并不天然等同于 `FlatBuffers`。如果强依赖 FlatBuffers，需要调整协议栈设计。

建议的方向：

- 控制面优先：`Tonic + Protobuf`
- 数据面优先：`Quinn` 或自定义传输层承载大块状态
- 若坚持 `FlatBuffers`，建议将其用于数据面 payload，而不是直接替换 gRPC 编码层

### 6.2 Pipeline Parallelism

- 定义 stage 切分策略
- 明确 prefill/decode 节点职责
- 设计节点间 backpressure 机制

### 6.3 Zero-Copy Migration

- 先定义状态块结构
- 再评估是否引入 RDMA
- 没有稳定 block 抽象之前，不应过早做跨节点迁移实现

## 7. 里程碑建议

### M0: Repo Bootstrap

- workspace 与 crate 初始化
- README / roadmap / detailed plan / CI 就位

### M1: Local First Token

- mmap 加载真实模型
- 动态装载一个最小 GEMM kernel
- 跑通真实 first token

### M2: Single-Node Scheduler

- block manager
- continuous batching
- tokenizer 异步化

### M3: CPU Optimization Pack

- SIMD dispatch
- quantization
- NUMA awareness

### M4: Distributed Prototype

- RPC
- prefill/decode 分离
- pipeline parallel

## 8. 风险与约束

- 过早进入分布式会放大尚未稳定的本地抽象
- 没有 benchmark 的性能优化容易把复杂度推高但收益不明
- 动态库 ABI 如果一开始不约束清楚，后续 kernel 生态会很难维护
- `unsafe` 代码和跨语言接口需要严格收口

## 9. 初始化后的下一步

最合理的下一个开发任务不是继续扩目录，而是立刻实现 M1：

1. 为 `GGUF` 建立最小元数据解析
2. 约定一个 C ABI 的 `AVX2 GEMM` kernel 接口
3. 在 `lattice-runtime` 中串起一次真实矩阵计算
4. 把占位 token 替换成真实 logits 解码输出
