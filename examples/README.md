# Examples

当前目录提供最小可运行示例，其中 `load_model.rs` 用于打通“自动下载模型 -> 映射权重 -> 解析 GGUF -> 构造 LLaMA 运行绑定”的链路。

这个示例当前验证的是 smoke path，而不是真实推理。仓库目前已经能完成模型下载、缓存复用、格式识别、`mmap` 映射、GGUF 元数据解析，以及 LLaMA 架构的关键 tensor 绑定，但还没有实现 tokenizer、算子执行、logits 计算和采样解码，所以输出不代表模型真的生成了 token。

## 如何实现

这次实现没有手写 `curl` 或自定义缓存逻辑，而是直接使用 Hugging Face 官方维护的 [`hf-hub`](https://crates.io/crates/hf-hub) crate。仓库根目录新增了一个 facade crate `lattice`，这样 `examples/load_model.rs` 可以直接通过 `use lattice::...` 调用模型目录解析和运行时入口。

核心实现分成四部分：

- `.gitignore` 新增 `models/` 和 `.hf_cache/`，避免本地下载产物被 Git 跟踪
- `src/models.rs` 提供 `load_model_catalog()`、`resolve_model()` 和 `resolve_model_from_catalog()`：
  - `load_model_catalog()` 负责解析根目录 `models.toml`
  - `resolve_model()` 负责将单个 `ModelSpec` 解析到本地缓存路径，必要时自动下载
  - `resolve_model_from_catalog()` 负责按名字从 catalog 中取出模型并完成解析
- `models.toml` 作为简单 model catalog，当前预置了 `smollm-small` 和 `llama3-basic`
- `examples/load_model.rs` 默认通过 catalog 解析 `llama3-basic`，拿到本地模型路径后交给 `InferenceRuntime::builder().with_model_path(...).build()`；运行时内部仍然沿用现有的 `mmap` 加载流程
- 映射完成后，示例会继续走 `MappedWeights::gguf_file()` 和 `LlamaModel::from_gguf()`，验证 GGUF 元数据、LLaMA 超参和关键 tensor 命名能被正确解析
- runtime 暴露的仍然是 `bootstrap_status()`，只用于说明权重已经映射、运行时已经就绪，不会执行真实模型前向

## 运行方式

```bash
cargo run --example load_model
```

如果你想显式指定 catalog 模型名，可以把模型名作为第一个参数传入：

```bash
cargo run --example load_model -- llama3-basic
```

首次运行时，`hf-hub` 会在系统缓存目录中检查并下载模型文件；后续运行会直接复用缓存，不需要重新拉取。当前示例默认读取：

```toml
[smollm-small]
repo = "bartowski/SmolLM2-135M-Instruct-GGUF"
file = "SmolLM2-135M-Instruct-Q8_0.gguf"
```

```toml
[llama3-basic]
repo = "bartowski/Llama-3.2-1B-Instruct-GGUF"
file = "Llama-3.2-1B-Instruct-Q8_0.gguf"
```

如果你想切换测试模型，只需要修改 `models.toml` 或新增一个别名，然后把模型名作为参数传给 `load_model.rs`。需要注意的是：当前 smoke path 只支持 `general.architecture = "llama"` 的 GGUF。

## 启动流程

1. 用户执行 `cargo run --example load_model -- llama3-basic`
2. 示例先读取 `models.toml`，解析出目标 Hugging Face repo 和文件名
3. `hf-hub` 检查本机缓存；如果本地不存在，则自动下载目标权重
4. 下载完成后，示例拿到本地文件路径并交给 `InferenceRuntime`
5. `InferenceRuntime` 内部用 `memmap2` 将 GGUF 文件映射进内存，并返回 bootstrap 就绪状态
6. 示例继续解析 GGUF metadata，构造 `LlamaModelSpec` 和 `LlamaTensorMap`
7. 打印关键超参与 tensor 绑定结果，确认“运行所需结构”已齐备

## 当前不包含的能力

下面这些能力在当前仓库里还没有实现，因此这个 example 不应该被视为“模型推理已跑通”：

- tokenizer 编码与解码
- attention / matmul / RMSNorm / RoPE 等真实前向算子
- KV cache 和 decode loop
- logits 到 token 的采样输出

这个设计的优点是：

- Git 仓库保持轻量，不携带大模型文件
- 新用户只需要安装 Rust 工具链即可运行最小示例
- Hugging Face 的全局缓存可以跨项目复用，避免重复占用磁盘
