"""Skills 数据结构 + 学习算法

核心思想:
- 每个 prompt 抽取一个 signature (如 "M|list/num")
- Skill 记录该 signature 下每个模型的 success/total 统计
- 基于统计决定"能不能用便宜模型"
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================================
# Prompt → Signature 提取
# ============================================================================

def extract_signature(prompt: str) -> str:
    """
    从 prompt 提取 signature (cluster key)。
    
    格式: "<length_bucket>|<tag1>/<tag2>/..."
    例如:
      "M|list/num"  中等长度 + 涉及 list 和 num
      "L|crypto/str"  长 + 密码学 + 字符串
      "S|general"   短 + 通用
    """
    p = prompt.lower()
    length = len(prompt)

    # 长度分桶
    if length < 200:
        len_b = "S"
    elif length < 500:
        len_b = "M"
    else:
        len_b = "L"

    # 标签提取 (基于关键词)
    tags = []
    if any(k in p for k in ["list", "array"]):
        tags.append("list")
    if any(k in p for k in ["string", "str"]):
        tags.append("str")
    if any(k in p for k in ["number", "integer", "float"]):
        tags.append("num")
    if any(k in p for k in ["sort", "order"]):
        tags.append("sort")
    if any(k in p for k in ["prime", "factor", "divisor"]):
        tags.append("theory")
    if any(k in p for k in ["encode", "decode", "cipher", "cyclic", "shift"]):
        tags.append("crypto")
    if any(k in p for k in ["poly", "recursion", "recursive"]):
        tags.append("advanced")
    if any(k in p for k in ["true", "false", "check", "boolean"]):
        tags.append("bool")

    if not tags:
        tags.append("general")

    return f"{len_b}|{'/'.join(sorted(set(tags)))}"


# ============================================================================
# Skill (一个 cluster 的统计)
# ============================================================================

class Skill:
    """
    对一个 signature cluster 记录每个模型的成功率统计。
    
    核心方法:
    - update(model, success): 记一笔新数据
    - model_success_rate(model): 返回该 model 对这类题的成功率
    - recommend_cheapest_viable_model(candidates, min_rate): 推荐最便宜可用模型
    """

    def __init__(self, signature: str):
        self.signature = signature
        # {model_id: [successes, total]}
        self.stats: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
        self.history: List[Dict] = []  # 完整事件记录

    def update(self, model_id: str, success: bool, task_id: str = ""):
        """记录一次调用结果"""
        self.stats[model_id][1] += 1
        if success:
            self.stats[model_id][0] += 1
        self.history.append({
            "task_id": task_id,
            "model": model_id,
            "success": success,
        })

    def model_success_rate(
        self, model_id: str, use_laplace: bool = True
    ) -> Tuple[float, int]:
        """
        返回 (success_rate, n_samples)。
        use_laplace=True 时用 Laplace 平滑 (避免小样本过估)。
        """
        s, t = self.stats[model_id]
        if t == 0:
            return 0.5 if use_laplace else 0.0, 0
        if use_laplace:
            return (s + 1) / (t + 2), t
        return s / t, t

    def recommend_cheapest_viable_model(
        self,
        candidates: List[str],  # 按价格从便宜到贵排序
        min_rate: float = 0.85,
        min_samples: int = 2,
    ) -> Optional[str]:
        """
        推荐最便宜但成功率 >= min_rate 的模型。
        
        Args:
            candidates: 模型 ID 列表 (按价格从便宜到贵排)
            min_rate: 最低可接受成功率 (Laplace 平滑后)
            min_samples: 至少要几个样本才能做推荐
        
        Returns:
            推荐的 model_id, 或 None (不确定)
        """
        for mid in candidates:
            rate, n = self.model_success_rate(mid, use_laplace=True)
            if n >= min_samples and rate >= min_rate:
                return mid
        return None

    def can_downgrade_to_small(
        self,
        small_model: str,
        min_rate: float = 0.8,
        min_samples: int = 1,
    ) -> Optional[bool]:
        """
        判断这个 cluster 能否降级到小模型？
        
        Returns:
            True  : 小模型能胜任 (省钱)
            False : 小模型不行, 必须升级
            None  : 数据不够, 无法判断
        """
        s, t = self.stats[small_model]
        if t < min_samples:
            return None
        return (s / t) >= min_rate

    def to_dict(self) -> Dict:
        """序列化为 JSON"""
        return {
            "signature": self.signature,
            "stats": {mid: list(v) for mid, v in self.stats.items()},
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Skill":
        """从 JSON 反序列化"""
        s = cls(data["signature"])
        s.stats = defaultdict(lambda: [0, 0])
        for mid, v in data["stats"].items():
            s.stats[mid] = list(v)
        s.history = data.get("history", [])
        return s


# ============================================================================
# SkillBook: 管理所有 skills
# ============================================================================

class SkillBook:
    """所有 skills 的容器 + 读写"""

    def __init__(self):
        self.skills: Dict[str, Skill] = {}

    def get_or_create(self, signature: str) -> Skill:
        """拿到 skill，不存在就新建"""
        if signature not in self.skills:
            self.skills[signature] = Skill(signature)
        return self.skills[signature]

    def update(self, prompt: str, model_id: str, success: bool, task_id: str = ""):
        """便利方法: 从 prompt 抽 signature 再更新"""
        sig = extract_signature(prompt)
        skill = self.get_or_create(sig)
        skill.update(model_id, success, task_id)

    def recommend(
        self,
        prompt: str,
        candidates: List[str],
        min_rate: float = 0.85,
        min_samples: int = 2,
    ) -> Tuple[Optional[str], Optional[Skill]]:
        """
        对一个新 prompt 给出推荐。
        
        Returns:
            (recommended_model_id, matched_skill)
        """
        sig = extract_signature(prompt)
        skill = self.skills.get(sig)
        if skill is None:
            return None, None
        rec = skill.recommend_cheapest_viable_model(candidates, min_rate, min_samples)
        return rec, skill

    def save(self, path: Path):
        """保存到 JSON"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {"skills": [s.to_dict() for s in self.skills.values()]},
                f,
                indent=2,
            )

    def load(self, path: Path):
        """从 JSON 加载"""
        path = Path(path)
        if not path.exists():
            return
        with open(path) as f:
            data = json.load(f)
        for s_data in data.get("skills", []):
            s = Skill.from_dict(s_data)
            self.skills[s.signature] = s

    def summary(self) -> Dict:
        """返回简要统计"""
        total = len(self.skills)
        return {
            "total_skills": total,
            "total_observations": sum(
                sum(v[1] for v in s.stats.values()) for s in self.skills.values()
            ),
            "signatures": list(self.skills.keys()),
        }
