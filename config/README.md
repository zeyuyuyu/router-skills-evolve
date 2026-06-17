# 实验配置（run_full_pipeline.sh 的输入存档）

每个 `*.env` 记录一次实验的**输入**（环境变量 + 等价 flags），既是存档也可直接重跑：

```bash
# 用存好的配置重跑（--config 接受裸名 / 路径 / 绝对路径）
bash scripts/run_full_pipeline.sh --config humaneval_dapo_gpt --bench humaneval --n-cycles 1
# 或
EXPERIMENT_CONFIG=humaneval_dapo_gpt bash scripts/run_full_pipeline.sh --bench humaneval
```

加载顺序：`.env`（密钥）→ `config/<name>.env`（实验配方）→ pipeline 内置默认。
**优先级**：内联传入的 env/flag > config 文件 > 默认值。

`--bench` / `--n-cycles` 仍走命令行 flag（不在 config 里），因为它们决定流程本身。

## 现有配置

| 文件 | distiller | RL | SFT 数据 | 推理后端 | 备注 |
|------|-----------|----|---------| --------|------|
| `humaneval_grpo.env` | heuristic | GRPO | hard-only | HF | 最早基线 |
| `humaneval_dapo.env` | heuristic | DAPO | hard-only | HF | DAPO 对照 |
| `humaneval_dapo_gpt.env` | gpt-5.4-mini | DAPO | +success | HF | 当前主力（修复 SFT 崩溃） |
| `humaneval_vllm_dapo_gpt.env` | gpt-5.4-mini | DAPO | +success | **vLLM** | 3-GPU 布局，待驱动支持 cu13 后启用 |

## vLLM 注意

`HE_USE_VLLM=1` / `GRPO_USE_VLLM=1` 当前在本机驱动（575 / CUDA 12.9）下无法运行：
vllm 0.22 硬依赖 flashinfer，flashinfer 0.6 是 cu13 kernel，超过驱动上限。
驱动升级到支持 CUDA 13 后，`humaneval_vllm_dapo_gpt.env` 即可直接用。
