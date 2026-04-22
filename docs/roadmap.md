# Lattice Roadmap

## 总体目标

Lattice 的演进方向分成四个阶段，每个阶段都必须满足两个标准：

- 有清晰的可验证交付物
- 对下一阶段的工程边界有增量复用价值

## L1: Core Foundations

目标：实现从磁盘加载模型到 CPU 吐出第一个 Token 的完整链路，验证 Rust 调用路径、内存布局和基础 kernel 接口的可行性。

### 核心交付物

- 基础 `Tensor` 数据结构
- 对齐的 CPU 内存分配器
- 可加载 `.so` 算子的 `KernelLauncher`
- 单 batch、同步推理主流程
- `GGUF` / `Safetensors` 权重的 `mmap` 映射

### 验收标准

- 能成功检测模型格式并映射权重文件
- 能加载一个外部动态库中的示例 kernel
- 能执行一条最小化解码流程并返回第一个 Token
- 全链路具备基本日志和错误处理

## L2: Scheduled Lattice

目标：引入 vLLM 风格的核心调度能力，在单机上支持高并发请求，并降低 Host-bound 开销。

### 核心交付物

- `Lattice Block Manager`
- `Continuous Batching`
- 面向 CPU cache 特性的 `PagedAttention`
- Rust 原生异步 tokenizer 集成

### 验收标准

- 新请求可在 decode 过程中动态加入队列
- KV Cache 能以固定 block 进行分配和回收
- 在并发请求下吞吐显著优于 L1 的串行流程
- 调度层和执行层接口完成解耦

## L3: Hardware Mastery

目标：针对不同 CPU 架构构建专用高性能 kernel，使单机性能逼近或超过 `llama.cpp` 同级实现。

### 核心交付物

- `AVX2` / `AVX-512` / `AMX` / `SVE` 专用 kernel
- `GGUF K-Quants` 优先的量化支持
- 运行时算子融合能力
- `NUMA` 感知调度

### 验收标准

- 常见矩阵算子具备面向目标架构的专用实现
- 量化权重推理具备可比较的 benchmark
- 多路 NUMA 机器上性能退化可控
- kernel 选择策略具备可观测性

## L4: Distributed Mesh

目标：跨越单机内存与吞吐限制，构建可横向扩展的分布式推理网络。

### 核心交付物

- `Prefill / Decode` 物理解耦
- 低延迟 RPC 通信层
- 跨节点 `Pipeline Parallelism`
- KV Cache 状态的高效迁移

### 验收标准

- 支持多节点模型分片与流水线执行
- 支持跨节点 transfer 的性能观测
- 具备最小可用的调度与容错策略
- 能在多台大内存机器上运行超单机容量模型

## 阶段依赖关系

- L2 依赖 L1 提供稳定的 tensor / weights / runtime 基础
- L3 依赖 L2 的执行接口稳定，否则优化对象会不断漂移
- L4 依赖 L2/L3 对内存、调度、kernel 的边界收敛

## 当前优先级

当前优先级应严格聚焦在 L1，不建议过早实现分布式控制面。先把 “本地加载 GGUF + 动态 kernel + first token” 做成一个可 benchmark 的闭环。
