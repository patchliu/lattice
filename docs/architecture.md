# Architecture

## 设计原则

- 数据结构边界先行
- 运行时与底层算子解耦
- 控制面与数据面最终分离
- 单机路径先稳定，再扩展分布式

## 模块依赖方向

```text
lattice-cli
    |
    v
lattice-runtime ---> lattice-kernels
    |                   |
    v                   v
lattice-model      external .so kernels
    |
    v
lattice-tensor ---> lattice-allocator
    ^
    |
lattice-core
```

## 模块职责

### lattice-core

定义全局错误类型、基础 Result 和共享类型，避免 workspace 内重复定义核心约定。

### lattice-allocator

负责对齐内存分配。当前阶段只提供基础 aligned buffer，后续可向 block allocator 演进。

### lattice-tensor

负责 tensor 的形状、dtype 和底层 buffer 封装。它不直接负责模型格式，也不直接负责任务调度。

### lattice-model

负责权重来源与映射，包括：

- 格式识别
- 零拷贝读取入口
- 后续元数据解析与 tensor 索引

### lattice-kernels

负责加载外部动态库并管理 kernel 入口。这里保持最小抽象，避免太早把调度策略、设备选择和 ABI 细节耦合进去。

### lattice-runtime

负责编排 L1 的主执行路径。未来 L2 的 scheduler 很可能会拆成独立 crate，但 runtime 仍应保留执行上下文与生命周期管理的职责。

## 未来扩展点

- `lattice-scheduler`: 请求编排、batch 合并、backpressure
- `lattice-tokenizer`: 编解码流水线
- `lattice-quant`: 量化权重与解压 kernel
- `lattice-distributed`: RPC、节点拓扑、迁移与容错

## unsafe 约束

当前架构中最可能出现 `unsafe` 的区域：

- 对齐内存分配
- `mmap` 映射
- 动态库与 FFI kernel 入口
- SIMD / 平台特化代码

这些区域应该被限制在少量 crate 中，并且必须把不变量写清楚。
