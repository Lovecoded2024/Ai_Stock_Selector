import time
from functools import wraps
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retry_on_exception(retries=3, delay=1):
    """
    重试装饰器，用于处理API调用失败的情况
    
    Args:
        retries (int): 最大重试次数
        delay (int): 重试间隔（秒）
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < retries - 1:  # 如果不是最后一次尝试
                        wait_time = delay * (attempt + 1)  # 递增等待时间
                        logger.warning(f"调用 {func.__name__} 失败 (尝试 {attempt + 1}/{retries}): {str(e)}")
                        logger.info(f"等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"调用 {func.__name__} 最终失败: {str(e)}")
            
            # 所有重试都失败后，返回错误信息
            error_msg = str(last_exception) if last_exception else "未知错误"
            if 'get_news_sentiment' in func.__name__:
                return f"无法获取新闻舆情数据: {error_msg}"
            elif 'get_detailed_analysis' in func.__name__:
                return {
                    '代码': args[0] if args else 'Unknown',
                    '技术面分析': '无法生成技术面分析',
                    '基本面分析': '无法生成基本面分析',
                    '新闻舆情': '无法获取新闻舆情数据',
                    '交易建议': '无法生成交易建议',
                    '风险提示': '无法生成风险提示'
                }
            elif 'get_llm_rating' in func.__name__:
                return {
                    'rating': 50,
                    'analysis': f'无法生成分析报告: {error_msg}',
                    'recommendation': '持有'
                }
            return None
            
        return wrapper
    return decorator
