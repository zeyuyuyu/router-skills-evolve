# 实验配置（run_full_pipeline.sh 的输入存档）

每个 `*.yaml` 用**扁平 `KEY: value`** 记录一次实验的输入(环境变量),既是存档也可直接重跑：

```bash
# 用存好的配置重跑(--config 接受裸名 / 路径 / 绝对路径,自动补 .yaml)
PYTHON=$PWD/venv/bin/python EXPERIMENT_CONFIG=humaneval_dapo_gpt \
  bash scripts/run_full_pipeline.sh --bench humaneval --n-cycles 1
# 或
bash scripts/run_full_pipeline.sh --config humaneval_dapo_gpt --bench humaneval --n-cycles 1
```

加载顺序：`.env`(密钥)→ `config/<name>.yaml`(实验配方)→ pipeline 内置默认。
**优先级**：内联传入的 env/flag > config 文件 > 默认值。

格式说明：扁平 `KEY: value`,`#` 注释,不支持嵌套。`--bench` / `--n-cycles` 仍走命令行
flag(不在 config 里),因为它们决定流程本身。旧的 `.env`(可 source 的 shell)仍兼容。

## 现有配置

| 文件 | bench | distiller | RL | SFT 后端 | vLLM | 备注 |
|------|-------|-----------|----|---------| -----|------|
| `humaneval_dapo_gpt.yaml` | humaneval | gpt-5.4-mini | DAPO | HF | 否(可选提速) | **当前主力**(修复 SFT 崩溃 + skill 用全部 trace) |
| `tau2_retail.yaml` | tau2_bench | deepseek-v3.2 | 关(opt-in) | FSDP2 | 否 | 远程 API small/large,默认这条路不碰 vLLM |

> 改实验直接复制这个 yaml 改字段。vLLM 版(`HE_USE_VLLM=1`/`GRPO_USE_VLLM=1` +
> GPU 布局 `HE_VLLM_SMALL_GPU`/`HE_VLLM_LARGE_GPU`)等驱动支持 cu13 后再加(见下)。

**两个 bench 的 vLLM 不是一回事**:HumanEval 的 `HE_USE_VLLM`/`GRPO_USE_VLLM` 是纯
*提速*开关(默认 0 = in-process HF,完全能跑);tau2 没有这个开关,vLLM 仅在 **cycle ≥ 1
且 small model 是本地训出的 checkpoint** 时自动启动(把 student serve 成 OpenAI endpoint,
无 HF fallback)。tau2 默认用远程 API small/large → 全程不碰 vLLM。

## 关键开关

| KEY | 作用 |
|-----|------|
| `DISTILLER_MODEL` | skill 蒸馏用的 LLM(`heuristic` 关闭) |
| `SKILL_MAX_EXEMPLARS` / `SKILL_DISTILL_N` | 每个 skill 保留 / 喂给 distiller 的 exemplar 数(默认 50;旧默认 8 太少,浪费 trace) |
| `SFT_INCLUDE_SUCCESS` | 1=也克隆已解出的题(避免 hard-only nan 梯度崩溃) |
| `GRPO_ALGO` | `grpo`(对称 clip) / `dapo`(非对称 + 动态采样) |
| `HE_USE_VLLM` / `GRPO_USE_VLLM` | vLLM 后端(本机驱动下须为 0,见下) |

## vLLM 注意

`HE_USE_VLLM=1` / `GRPO_USE_VLLM=1` 当前在本机驱动(575 / CUDA 12.9)下无法运行：
vllm 0.22 硬依赖 flashinfer,flashinfer 0.6 是 cu13 kernel,超过驱动上限。
驱动升级到支持 CUDA 13 后,在 yaml 里加上 `HE_USE_VLLM: 1` / `GRPO_USE_VLLM: 1`
+ GPU 布局即可启用(并用 vLLM-capable venv 跑,见 `scripts/setup_vllm_venv.sh`)。
