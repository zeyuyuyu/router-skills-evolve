"""配置: API keys, 模型池, 价格"""

import os

# ============================================================================
# API Keys
# ============================================================================

COMMONSTACK_API_KEY = os.environ.get("COMMONSTACK_API_KEY", "")


def require_api_key():
    """在需要 API 调用时调用此函数, 检查 key 是否设置"""
    if not COMMONSTACK_API_KEY:
        raise RuntimeError(
            "COMMONSTACK_API_KEY 未设置!\n"
            "请运行: export COMMONSTACK_API_KEY='your-key'\n"
            "或在代码开头: os.environ['COMMONSTACK_API_KEY'] = '...'"
        )

COMMONSTACK_BASE_URL = "https://api.commonstack.ai/v1"
PROXY_BASE_URL = "http://127.0.0.1:8403/v1"  # UncommonRoute proxy

# ============================================================================
# 模型池 (按价格从便宜到贵排序)
# ============================================================================

# 每个模型: (prompt_price, completion_price) USD per 1M tokens
MODEL_POOL = {
    # 小模型 (便宜)
    "deepseek/deepseek-v3.2": {
        "label": "SMALL",
        "price": (0.269, 0.40),
        "open_source": True,  # 可训练
        "params": "37B activate (671B MoE)",
    },
    # 大模型 (贵但强)
    "openai/gpt-5.4-2026-03-05": {
        "label": "LARGE",
        "price": (2.50, 15.0),
        "open_source": False,
        "params": "?",
    },
}

# 简化别名
SMALL_MODEL = "deepseek/deepseek-v3.2"
LARGE_MODEL = "openai/gpt-5.4-2026-03-05"

# 动态模型价格查找表 (用于 proxy 返回任意模型时计算成本)
DYNAMIC_PRICES = {
    # OpenAI (闭源)
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4.1": (2.0, 8.0),
    "openai/gpt-5": (1.25, 10.0),
    "openai/gpt-5.2": (1.75, 14.0),
    "openai/gpt-5.3-codex": (1.75, 14.0),
    "openai/gpt-5.4-2026-03-05": (2.50, 15.0),
    "openai/gpt-5.4-mini-2026-03-17": (0.75, 4.50),
    "openai/gpt-5.4-nano-2026-03-17": (0.20, 1.25),
    "openai/gpt-5.4-pro-2026-03-05": (30.0, 180.0),
    "openai/gpt-oss-120b": (0.05, 0.25),  # 开源!

    # DeepSeek (开源)
    "deepseek/deepseek-v3.2": (0.269, 0.40),
    "deepseek/deepseek-v3.1": (0.27, 1.0),
    "deepseek/deepseek-r1-0528": (0.70, 2.50),
    "deepseek/deepseek-chat": (0.28, 0.42),

    # Anthropic (闭源)
    "anthropic/claude-haiku-4-5": (1.0, 5.0),
    "anthropic/claude-sonnet-4-5": (3.0, 15.0),
    "anthropic/claude-sonnet-4-6": (3.0, 15.0),
    "anthropic/claude-opus-4-5": (5.0, 25.0),
    "anthropic/claude-opus-4-6": (5.0, 25.0),
    "anthropic/claude-opus-4-7": (5.0, 25.0),

    # Google (闭源)
    "google/gemini-2.5-flash": (0.30, 2.50),
    "google/gemini-2.5-flash-image": (0.30, 2.50),
    "google/gemini-2.5-pro": (1.25, 10.0),
    "google/gemini-3-flash-preview": (0.30, 2.50),
    "google/gemini-3-pro-image-preview": (2.0, 12.0),
    "google/gemini-3.1-flash-image-preview": (0.15, 30.0),
    "google/gemini-3.1-flash-lite-preview": (0.25, 1.50),
    "google/gemini-3.1-pro-preview": (2.0, 12.0),

    # MiniMax (开源)
    "minimax/minimax-m2": (0.30, 1.20),
    "minimax/minimax-m2.1": (0.30, 1.20),
    "minimax/minimax-m2.5": (0.30, 1.20),
    "minimax/minimax-m2.7": (0.30, 1.20),

    # Zhipu GLM (开源)
    "zai-org/glm-4.5-air": (0.13, 0.85),
    "zai-org/glm-4.6": (0.60, 2.20),
    "zai-org/glm-4.7": (0.60, 2.20),
    "zai-org/glm-5": (1.0, 3.20),
    "zai-org/glm-5-turbo": (1.20, 4.0),
    "zai-org/glm-5.1": (1.40, 4.40),

    # Qwen (开源)
    "qwen/qwen3-coder-480b-a35b-instruct": (0.33, 1.65),
    "qwen/qwen3-vl-235b-a22b-instruct": (0.30, 1.50),
    "qwen/qwen3.5-397b-a17b": (0.60, 3.60),

    # Moonshot/Kimi (开源)
    "moonshotai/kimi-k2.5": (0.66, 3.30),
    "moonshotai/kimi-k2-0905": (0.60, 2.50),
    "moonshotai/kimi-k2-thinking": (0.60, 2.50),

    # xAI (闭源 API)
    "x-ai/grok-4-1-fast-non-reasoning": (0.20, 0.50),
    "x-ai/grok-4.1-fast-reasoning": (0.20, 0.50),
    "x-ai/grok-code-fast-1": (0.20, 1.50),

    # Xiaomi (开源)
    "xiaomi/mimo-v2-pro": (1.0, 3.0),
    "xiaomi/mimo-v2-omni": (0.40, 2.0),
}

# ============================================================================
# 开源模型列表 (可训练)
# ============================================================================

TRAINABLE_MODELS = {
    # 推荐训练候选 (按价格 + GPU 需求)
    # 训练难度: EASY (< 30B), MEDIUM (30-70B), HARD (> 70B)

    "minimax/minimax-m2": "EASY (~20B, 2xA100)",
    "minimax/minimax-m2.1": "EASY (~20B, 2xA100)",
    "minimax/minimax-m2.5": "EASY (~20B, 2xA100)",
    "minimax/minimax-m2.7": "EASY (~20B, 2xA100)",
    "zai-org/glm-4.5-air": "EASY (~12B MoE, 2xA100)",
    "xiaomi/mimo-v2-omni": "EASY (~10B, 1xA100)",

    "deepseek/deepseek-v3.2": "MEDIUM (37B activate, 8xA100)",
    "deepseek/deepseek-v3.1": "MEDIUM (37B activate, 8xA100)",
    "deepseek/deepseek-r1-0528": "MEDIUM (37B activate, 8xA100)",
    "deepseek/deepseek-chat": "MEDIUM (~30B, 4xA100)",
    "qwen/qwen3-coder-480b-a35b-instruct": "MEDIUM (35B activate, 8xA100)",
    "qwen/qwen3-vl-235b-a22b-instruct": "MEDIUM (22B activate, 4xA100)",
    "qwen/qwen3.5-397b-a17b": "MEDIUM (17B activate, 4xA100)",
    "zai-org/glm-4.6": "MEDIUM (~50B, 4xA100)",
    "zai-org/glm-4.7": "MEDIUM (~50B, 4xA100)",
    "zai-org/glm-5-turbo": "MEDIUM (~30B, 4xA100)",
    "xiaomi/mimo-v2-pro": "MEDIUM (~40B, 4xA100)",

    "moonshotai/kimi-k2.5": "HARD (~70B, 8xA100)",
    "moonshotai/kimi-k2-thinking": "HARD (~70B, 8xA100)",
    "moonshotai/kimi-k2-0905": "HARD (~70B, 8xA100)",
    "zai-org/glm-5": "HARD (~100B, 8xA100)",
    "zai-org/glm-5.1": "HARD (~100B, 8xA100)",
    "openai/gpt-oss-120b": "HARD (120B, 8xA100)",
}


def get_model_hf_name(model_id: str) -> str:
    """CommonStack model ID → HuggingFace 模型名（用于下载权重）"""
    mapping = {
        "deepseek/deepseek-v3.2": "deepseek-ai/DeepSeek-V3.2",
        "deepseek/deepseek-v3.1": "deepseek-ai/DeepSeek-V3",
        "deepseek/deepseek-r1-0528": "deepseek-ai/DeepSeek-R1",
        "qwen/qwen3-coder-480b-a35b-instruct": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
        "qwen/qwen3.5-397b-a17b": "Qwen/Qwen3.5-397B-A17B",
        "moonshotai/kimi-k2.5": "moonshotai/Kimi-K2-Instruct",
        "zai-org/glm-4.5-air": "zai-org/GLM-4.5-Air",
        "zai-org/glm-4.7": "zai-org/GLM-4.7",
        "minimax/minimax-m2": "MiniMaxAI/MiniMax-M2",
        "minimax/minimax-m2.5": "MiniMaxAI/MiniMax-M2.5",
        "openai/gpt-oss-120b": "openai/gpt-oss-120b",
    }
    return mapping.get(model_id, model_id)


def calc_cost(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    """计算成本 (USD)"""
    price = DYNAMIC_PRICES.get(model_id, (2.0, 10.0))  # fallback
    return (prompt_tokens / 1e6 * price[0] + 
            completion_tokens / 1e6 * price[1])
