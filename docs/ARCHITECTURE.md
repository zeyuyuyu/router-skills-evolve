# 架构说明

## 整体 Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│   用户 Prompt                                                │
│       │                                                       │
│       ▼                                                       │
│   ┌──────────────────┐   本地计算 (<10ms)                   │
│   │  UncommonRoute   │   无 LLM 调用                        │
│   │   Classifier     │   输出: tier (S/M/L)                 │
│   └────────┬─────────┘                                       │
│            │ tier                                            │
│            ▼                                                 │
│   ┌──────────────────┐   本地查询                           │
│   │   SkillBook      │   根据 prompt signature              │
│   │   (本项目核心)   │   推荐最优 model                     │
│   └────────┬─────────┘                                       │
│            │ final_model                                     │
│            ▼                                                 │
│   ┌──────────────────┐   真实 LLM 调用 (唯一成本)           │
│   │   LLM Call       │   通过 CommonStack API               │
│   │   (CommonStack)  │                                       │
│   └────────┬─────────┘                                       │
│            │ generated code                                  │
│            ▼                                                 │
│   ┌──────────────────┐   本地执行 (pytest)                  │
│   │   Code Test      │   验证是否通过                       │
│   └────────┬─────────┘                                       │
│            │ success/fail                                    │
│            ▼                                                 │
│   ┌──────────────────┐   收 trace → 学 skills               │
│   │   Skills Update  │   下次更聪明                         │
│   │   (Evolve)       │                                       │
│   └──────────────────┘                                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 核心模块

### 1. `src/config.py` - 配置中心

- API keys
- 模型池 (Small / Large)
- 价格表 (48 个模型)
- 开源模型列表 (24 个)

### 2. `src/models.py` - LLM 调用

```python
solve_task(model_id, task)
    → {"success": bool, "cost": float, "generated_code": str, ...}
```

一步完成: 调 LLM + 提取代码 + 跑测试。

### 3. `src/skills.py` - Skills 数据结构

**核心**: `Skill` 类记录每个 (prompt signature, model) 的成功率统计。

```python
skill = Skill("M|list/num")
skill.update("deepseek/deepseek-v3.2", success=True)
skill.can_downgrade_to_small(...)  # True / False / None
```

### 4. `src/router.py` - Router + Skills 路由

```python
router = RouterWithSkills(small_model, large_model, skill_book)
result = router.solve(task)
# 自动: 决策 → 调模型 → fallback → 更新 skills
```

## 关键设计

### Signature: prompt → cluster

用简单规则把 prompt 聚类到 signature：

```python
extract_signature("Write a function to sort a list of numbers")
# → "M|list/num/sort"
```

格式: `<length>|<tags>`
- Length: S (<200), M (200-500), L (>500)
- Tags: list, str, num, sort, crypto, advanced, bool 等

### Laplace 平滑 (避免小样本过估)

```python
success_rate = (successes + 1) / (total + 2)
```

新 skill 初始 rate = 0.5 (既不鼓励也不阻止降级)。

### Escalation + Fallback

1. 先试推荐模型 (可能是便宜的)
2. 若失败 → 升级到 large model
3. 保证 **最终准确率 >= 大模型独跑**

### Evolve 三轨

| 轨道 | 每次做什么 | 频率 |
|------|----------|------|
| Skills Evolve | 收 trace → 更新 skill 统计 | 每次请求 |
| Model Evolve | 从 traces 训练小模型 | 每周一次 |
| Router Evolve | UncommonRoute bandit 自调整 | 每次请求 |

## 成本说明

**Router + Skills 本身 0 成本**:
- UncommonRoute classifier: 本地 ML 模型 (~10ms)
- Skills 查询: Python 字典查找 (<1ms)
- **只有 LLM 调用才花钱**

这就是为什么 Skills 能省钱：**用便宜的模型代替贵的**。
