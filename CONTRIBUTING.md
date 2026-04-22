# Contributing

## 开发原则

- 先保证模块边界清晰，再追求实现速度
- 所有性能优化都应绑定可复现的 benchmark
- 未经验证的底层优化不要直接合入主分支
- 所有 unsafe 代码必须说明边界条件和不变量

## 本地开发

```bash
cargo fmt --all
cargo clippy --workspace --all-targets
cargo test --workspace
```

## 提交建议

- 小步提交，避免把架构重构和性能优化混在一个 PR
- 为 crate 级别的改动补充对应文档
- 修改 memory layout、ABI、kernel 接口时，务必同步更新架构文档

## 代码组织约定

- 通用类型放在 `lattice-core`
- 内存与 buffer 管理放在 `lattice-allocator`
- Tensor 结构放在 `lattice-tensor`
- 权重读取与格式解析放在 `lattice-model`
- 调度和 runtime 编排放在 `lattice-runtime`
- 动态算子装载与调用放在 `lattice-kernels`

## 未来约定

- L2 开始引入异步调度后，新增 crate 应明确区分 control plane 与 data plane
- L3/L4 的平台专用代码需要按 CPU 特性和传输协议拆目录，避免在同一模块里堆叠条件编译
