# LLaMA Prefill Plan

## Goal

在不引入 tokenizer、KV cache、采样器和外部 kernel ABI 的前提下，先跑通一条最小但真实的 LLaMA prefill 链路：

`token ids -> embedding -> transformer blocks -> output norm -> lm head logits -> argmax`

这条链路的目标不是性能，而是 correctness 和 runtime 边界稳定性。只要这条路径能稳定给出可验证的 next-token 结果，后续 tokenizer、采样器、KV cache 和 SIMD kernel 都可以沿着同一个执行骨架继续演进。

## Scope

这次实现明确只覆盖以下内容：

- 输入是已经准备好的 `token ids`
- 执行完整 prefill，并计算最后一个位置的 logits
- 输出 `argmax` token id 和完整 logits
- 支持 GGUF 中 first-token 所需的最小 tensor decode 能力

这次实现明确不做：

- tokenizer 编码与解码
- KV cache
- incremental decode loop
- top-k / top-p / temperature 等采样
- 外部 `.so` kernel
- 多 batch / continuous batching

## Runtime Placement

LLaMA prefill 逻辑放在 `lattice-runtime`，而不是 `lattice-model`：

- `lattice-model` 负责 GGUF 解析、tensor 绑定和模型结构校验
- `lattice-runtime` 负责执行编排和前向

因此公共入口建立在 `InferenceRuntime::load_llama()` 之后，由 `LoadedLlama` 暴露推理接口：

- `prefill(token_ids)`

这样可以保持“模型加载”和“模型执行”的职责边界清晰。

## Supported Tensor Types

为了让当前 catalog 里的真实模型尽快可跑，这版 prefill path 直接支持：

- `F32`
- `F16`
- `Q8_0`

优先支持这三类的原因：

- 测试用 tiny 模型最容易用 `F32`
- 真实小模型里 norm/少量权重常见 `F16`
- 当前 catalog 里的 `smollm-small` 和 `llama3-basic` 都是 `Q8_0` GGUF

K-Quants 暂时只保留 parser 支持，不纳入 first-token 执行范围。

## Execution Pipeline

对输入 prompt 的所有 token 做一次完整 prefill，然后只取最后一个位置的输出：

1. 从 `token_embd.weight` 读取每个 token 的 embedding，得到 `hidden_states[seq_len][embedding]`
2. 对每个 transformer block 依次执行：
   - attention RMSNorm
   - `attn_q / attn_k / attn_v` 线性投影
   - RoPE
   - causal self-attention
   - `attn_output` 投影并 residual add
   - FFN RMSNorm
   - `ffn_gate / ffn_up / ffn_down`
   - SiLU-gated MLP
   - residual add
3. 对最后一层输出做 `output_norm`
4. 用 `output` 或 tied embeddings 计算 logits
5. 对 logits 做 `argmax`

## API Shape

建议暴露一个最小返回结构：

- `next_token_id`: argmax 结果
- `logits`: 最后一个位置的完整 logits

这样后续接 sampler 时不需要推翻接口。

## Code Structure

代码按两层拆开：

### 1. LLaMA Orchestration

- `llama/`
- 对外接口：`LoadedLlama::prefill(&[u32]) -> Result<PrefillOutput>`
- 负责编排 embedding、block 执行、output norm 和 logits

### 2. Primitive Ops

- `ops/`
- `tensor.rs`: tensor row decode / shape helpers
- `matmul.rs`: linear / dot
- `norm.rs`: RMSNorm
- `rope.rs`: rotary embedding
- `attention.rs`: causal attention
- `activations.rs`: SiLU / SwiGLU

这样做的原因是：

- 编排层表达模型结构
- ops 层表达数值逻辑
- tensor decode 层表达 GGUF 存储细节

后续替换成 SIMD kernel 时，可以保留 `llama/` 编排层不动，逐步替换 `ops/` 实现。

## Validation Plan

验证分两层：

### Unit / Tiny GGUF

构造一个可控的 tiny GGUF：

- block 权重全零
- residual path 保留 embedding
- tied embeddings 输出

这样可以精确断言：

- 前向确实执行了完整链路
- 最终 argmax 等于预期 token

### Real Model Smoke

对 `smollm-small` 跑真实 `token ids` 输入，确认：

- 模型能完成 prefill
- logits 长度等于 vocab size
- argmax 能返回合法 token id

即使这一步还没有 tokenizer，也足以证明“真实 first token”路径已经建立。

## Non-Goals For This Iteration

这版实现不追求：

- 高性能
- 低内存占用
- decode 阶段复用
- quantized fused kernels

只要代码结构清晰、数值流程正确、接口可扩展，就达到了这一阶段的目标。
