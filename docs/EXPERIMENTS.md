# 已做实验结果汇总

## 最终实验: HumanEval 全量 164 题（真实 LLM + 真实代码测试）

### 5 种策略对比

| # | 策略 | 成功率 | 总成本 | 平均尝试 | vs Ceiling |
|---|------|--------|--------|---------|-----------|
| 1 | Pure GPT-5.4 (Ceiling) | 95% | $0.3655 | 1.00 | 100% |
| 2 | Pure DeepSeek (Floor) | 88% | $0.0142 | 1.00 | 4% |
| 3 | Router Only (auto) | 74% | $0.1535 | 1.00 | 42% |
| 4 | Router + Fallback | 98% | $0.2661 | 1.24 | 73% |
| 5 | 🏆 **Router + Skills Evolve** | **99%** | **$0.0638** | **1.07** | **17%** |

### Evolve 4 轮进化

| Round | Skills (start→end) | 成功率 | 单题成本 |
|-------|-------------------|--------|---------|
| 0 | 0→21 | 100% | $0.00037 |
| 1 | 21→33 | 100% | $0.00036 |
| 2 | 33→45 | 100% | $0.00041 |
| 3 | 45→49 | 95% | $0.00042 |

### 最终学到的 49 个 Skills

| 判断 | 数量 | 说明 |
|------|------|------|
| ✅ 便宜模型 OK | 40 | 可降级到 DeepSeek |
| ❌ 必须升级 | 9 | 必须用 GPT-5.4 |
| ? 数据不足 | 0 | - |

#### ❌ 9 个必须升级的 pattern

| Signature | 原因 |
|-----------|------|
| `L|advanced/list/num` | 多项式类 |
| `L|crypto/str` | 密码学 |
| `L|list/num` | 复杂 list/num 混合 |
| `L|list/str` | 长 list+string |
| `M|advanced/list` | 高级列表操作 |
| `M|bool/num` | 布尔数字 edge case |
| `M|bool/str` | 布尔字符串 |
| `M|list/num/sort` | 排序 + 数字 |
| `M|list/num/str` | 三类型混合 |

**这 9 个就是训练小模型的目标** —— 让它们能做好！

## 规模化预估

### 每月 1000 万请求的成本

| 策略 | 月度 | 年度 |
|------|------|------|
| Pure GPT-5.4 | $22,286 | $267,432 |
| Router + Fallback | $16,226 | $194,712 |
| **Router + Skills Evolve** | **$3,888** | **$46,656** |

**Evolve 年省 $220,776** vs Pure GPT-5.4。

## 之前的 8 个实验

| # | 实验 | 关键结论 |
|---|------|---------|
| 1 | HumanEval 10 题 | DeepSeek 100%, GPT-5.4 100% (简单题打平) |
| 2 | SWE-Bench 模拟 | Skills 有帮助但模拟不能当真 |
| 3 | Evolve Loop v1 (历史 traces) | Accuracy 67% → 94% |
| 4 | Naive Evolve (60 题) | 准确率 100% → 80% (过拟合) |
| 5 | Smart Evolve (4 机制) | 100% 但太保守，成本高 |
| 6 | HumanEval 1-30 控制变量 | Evolve = Pure DeepSeek (benchmark 过简单) |
| 7 | HumanEval 30-90 | Evolve 100%, 省 82% (真正混合难度)|
| 8 | **HumanEval 全量 164** | **Evolve 99%, 省 83%** (最终) |

详细报告见 `../uncommonroute_skill_experiment/`.
