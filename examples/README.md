# Examples

当前目录提供最小可运行示例，其中 `load_model.rs` 用于打通“自动下载模型 -> 映射权重 -> 进入运行时”的链路。

这个示例当前验证的是 bootstrap/load path，而不是真实推理。仓库目前已经能完成模型下载、缓存复用、格式识别和 `mmap` 映射，但还没有实现 GGUF 元数据解析、tokenizer、算子执行、logits 计算和采样解码，所以输出不代表模型真的生成了 token。

## 如何实现

这次实现没有手写 `curl` 或自定义缓存逻辑，而是直接使用 Hugging Face 官方维护的 [`hf-hub`](https://crates.io/crates/hf-hub) crate。仓库根目录新增了一个 facade crate `lattice`，这样 `examples/load_model.rs` 可以直接通过 `use lattice::...` 调用模型目录解析和运行时入口。

核心实现分成四部分：

- `.gitignore` 新增 `models/` 和 `.hf_cache/`，避免本地下载产物被 Git 跟踪
- `src/models.rs` 提供 `load_model_catalog()`、`resolve_model()` 和 `resolve_model_from_catalog()`：
  - `load_model_catalog()` 负责解析根目录 `models.toml`
  - `resolve_model()` 负责将单个 `ModelSpec` 解析到本地缓存路径，必要时自动下载
  - `resolve_model_from_catalog()` 负责按名字从 catalog 中取出模型并完成解析
- `models.toml` 作为简单 model catalog，当前预置了 `smollm-small` 和 `llama3-basic`
- `examples/load_model.rs` 通过 catalog 解析 `smollm-small`，拿到本地模型路径后交给 `InferenceRuntime::builder().with_model_path(...).build()`；运行时内部仍然沿用现有的 `mmap` 加载流程，因此下载完成后会直接映射权重文件
- runtime 暴露的是 `bootstrap_status()`，只用于说明权重已经映射、运行时已经就绪，不会执行真实模型推理

## 运行方式

```bash
cargo run --example load_model
```

首次运行时，`hf-hub` 会在系统缓存目录中检查并下载模型文件；后续运行会直接复用缓存，不需要重新拉取。当前示例默认读取：

```toml
[smollm-small]
repo = "bartowski/SmolLM2-135M-Instruct-GGUF"
file = "SmolLM2-135M-Instruct-Q8_0.gguf"
```

如果你想切换测试模型，只需要修改 `models.toml` 或新增一个别名，然后把 `load_model.rs` 中的模型名从 `smollm-small` 改成新的条目。

## 启动流程

1. 用户执行 `cargo run --example load_model`
2. 示例先读取 `models.toml`，解析出目标 Hugging Face repo 和文件名
3. `hf-hub` 检查本机缓存；如果本地不存在，则自动下载目标权重
4. 下载完成后，示例拿到本地文件路径并交给 `InferenceRuntime`
5. `InferenceRuntime` 内部用 `memmap2` 将 GGUF 文件映射进内存，并返回 bootstrap 就绪状态

## 当前不包含的能力

下面这些能力在当前仓库里还没有实现，因此这个 example 不应该被视为“模型推理已跑通”：

- GGUF 元数据和 tensor 索引解析
- tokenizer 编码与解码
- attention / matmul / RMSNorm / RoPE 等真实前向算子
- KV cache 和 decode loop
- logits 到 token 的采样输出

这个设计的优点是：

- Git 仓库保持轻量，不携带大模型文件
- 新用户只需要安装 Rust 工具链即可运行最小示例
- Hugging Face 的全局缓存可以跨项目复用，避免重复占用磁盘
