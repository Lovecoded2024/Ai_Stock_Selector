"""
配置文件,存储全局配置参数
支持多种API和模型配置
"""

# 默认配置
DEFAULT = {
    # LLM API配置
    "llm": {
        "deepseek": {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "",
            "models": {
                "rating": "deepseek/deepseek-chat",      # 用于日常评分的模型
                "analysis": "deepseek/deepseek-chat"  # 用于生成深度报告的模型
            }
        },
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "api_key": "",  # 需要填入OpenAI的API key
            "models": {
                "rating": "gpt-3.5-turbo",      # 用于日常评分的模型
                "analysis": "gpt-4"             # 用于生成深度报告的模型
            }
        },
         "gemini-2.0-flash-001": {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "",
            "models": {
                "rating": "google/gemini-2.0-flash-001",      # 用于日常评分的模型
                "analysis": "google/gemini-2.0-flash-001"  # 用于生成深度报告的模型
            }
        },       
    },
    
    # 线程池配置
    "thread_pool": {
        "max_workers": 50  # 可以根据需要调整线程数
    }
}

# 当前使用的配置(默认使用deepseek)
CURRENT = "gemini-2.0-flash-001"

# 导出实际使用的配置
LLM_CONFIG = DEFAULT["llm"][CURRENT]
MODEL_CONFIG = {
    "rating_model": LLM_CONFIG["models"]["rating"],
    "analysis_model": LLM_CONFIG["models"]["analysis"]
}
THREAD_POOL_CONFIG = DEFAULT["thread_pool"]
