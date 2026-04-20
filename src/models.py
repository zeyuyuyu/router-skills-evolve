"""LLM 调用封装 + HumanEval 测试"""

import signal
import time
from typing import Dict

from openai import OpenAI

from .config import (
    COMMONSTACK_API_KEY, COMMONSTACK_BASE_URL, PROXY_BASE_URL, calc_cost
)


# 两个 client:
# 1. 直接调 CommonStack (普通模型)
# 2. 通过 proxy (uncommon-route/auto 等)
direct_client = OpenAI(api_key=COMMONSTACK_API_KEY, base_url=COMMONSTACK_BASE_URL)
proxy_client = OpenAI(api_key=COMMONSTACK_API_KEY, base_url=PROXY_BASE_URL)


# ============================================================================
# 代码提取
# ============================================================================

def extract_code(response: str, entry_point: str, original_prompt: str = "") -> str:
    """
    从 LLM 响应中提取 Python 代码。
    
    处理 3 种情况:
    1. 完整函数 (含 def): 直接用
    2. Markdown 代码块 (```python...```): 提取
    3. 只有函数体 (无 def): 拼接原 prompt
    """
    # 提取 markdown 代码块
    if "```python" in response:
        start = response.index("```python") + len("```python")
        end = response.index("```", start)
        code = response[start:end].strip()
    elif "```" in response:
        start = response.index("```") + 3
        end = response.index("```", start)
        code = response[start:end].strip()
    else:
        code = response.strip()

    # 判断是完整函数还是只有函数体
    if f"def {entry_point}" in code:
        return code

    # 只有函数体 → 拼接 prompt + 函数体
    if original_prompt:
        body_lines = code.split("\n")
        indented = []
        for line in body_lines:
            if line.strip():
                if not line.startswith((" ", "\t")):
                    indented.append("    " + line)
                else:
                    indented.append(line)
            else:
                indented.append(line)
        return original_prompt + "\n".join(indented)

    return code


# ============================================================================
# LLM 调用
# ============================================================================

def call_llm(
    model_id: str,
    prompt: str,
    use_proxy: bool = False,
    temperature: float = 0.2,
    max_tokens: int = 1500,
) -> Dict:
    """
    调用 LLM，返回标准化结果。

    Args:
        model_id: 模型 ID, e.g. "deepseek/deepseek-v3.2" or "uncommon-route/auto"
        prompt: 用户 prompt
        use_proxy: True 走 UncommonRoute proxy（用于 uncommon-route/* 模型）
        temperature: 采样温度
        max_tokens: 最大输出

    Returns:
        {
          "response": str,       # 文本响应
          "code": str,           # 提取的代码（需要后续 extract）
          "actual_model": str,   # 实际调用的模型
          "prompt_tokens": int,
          "completion_tokens": int,
          "total_tokens": int,
          "cost_usd": float,
          "latency": float,
          "error": str,          # 错误信息 (若有)
        }
    """
    client = proxy_client if use_proxy else direct_client
    start = time.time()

    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content or ""
        # 有些模型 (minimax) 会把内容放在 reasoning_content
        if not text:
            text = getattr(resp.choices[0].message, "reasoning_content", "") or ""

        usage = resp.usage
        actual_model = resp.model
        cost = calc_cost(actual_model, usage.prompt_tokens, usage.completion_tokens)

        return {
            "response": text,
            "actual_model": actual_model,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "cost_usd": cost,
            "latency": time.time() - start,
            "error": "",
        }
    except Exception as e:
        return {
            "response": "",
            "actual_model": model_id,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0,
            "latency": time.time() - start,
            "error": str(e)[:200],
        }


# ============================================================================
# 代码测试 (HumanEval 风格)
# ============================================================================

class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Execution timeout")


def run_humaneval_test(task: Dict, generated_code: str, timeout: int = 10) -> tuple:
    """
    运行 HumanEval 测试。
    
    Args:
        task: {"entry_point": "func_name", "test": "test code with check()"}
        generated_code: 生成的 Python 代码
        timeout: 超时秒数

    Returns:
        (success: bool, error_message: str)
    """
    entry_point = task["entry_point"]
    test_code = task["test"]

    full_code = f"""
{generated_code}

{test_code}

check({entry_point})
"""

    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)

        namespace = {}
        exec(full_code, namespace)

        signal.alarm(0)
        return True, ""
    except AssertionError as e:
        return False, f"AssertionError: {str(e)[:100]}"
    except TimeoutError:
        return False, "Timeout"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:100]}"
    finally:
        signal.alarm(0)


# ============================================================================
# 高层 API: 调 LLM + 测试 = 一步搞定
# ============================================================================

def solve_task(model_id: str, task: Dict, use_proxy: bool = False) -> Dict:
    """
    完整的"解题"过程: 调模型 → 提取代码 → 运行测试。

    Args:
        model_id: 模型 ID
        task: HumanEval 格式 {"prompt", "entry_point", "test"}
        use_proxy: 走 UncommonRoute proxy 吗

    Returns:
        完整结果 dict (包含 success, cost, tokens 等)
    """
    prompt = (
        "Complete the following Python function. "
        "Only return the code, no explanations.\n\n"
        f"{task['prompt']}"
    )
    
    result = call_llm(model_id, prompt, use_proxy=use_proxy)
    
    if result["error"]:
        return {
            **result,
            "task_id": task.get("task_id", ""),
            "success": False,
            "generated_code": "",
        }
    
    code = extract_code(result["response"], task["entry_point"], task["prompt"])
    success, test_error = run_humaneval_test(task, code)
    
    return {
        **result,
        "task_id": task.get("task_id", ""),
        "success": success,
        "generated_code": code,
        "test_error": test_error,
    }
