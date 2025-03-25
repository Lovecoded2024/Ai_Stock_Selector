import akshare as ak
import datetime
import pandas as pd
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from utils import retry_on_exception
from config import LLM_CONFIG, THREAD_POOL_CONFIG

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置LLM
client = OpenAI(
    base_url=LLM_CONFIG["base_url"],
    api_key=LLM_CONFIG["api_key"]
)

# 创建线程池
thread_pool = ThreadPoolExecutor(max_workers=THREAD_POOL_CONFIG["max_workers"])

# 新闻分析prompt模板
NEWS_ANALYSIS_PROMPT = """
你是一位专业的股票分析师，请对以下股票新闻进行深入分析：

{news_content}

请按照以下专业格式返回分析结果：

1. 新闻概况：
- 新闻数量：{news_count}条
- 时间范围：最近7天
- 新闻来源：东方财富

2. 情绪状态判断：
- 总体情绪：[积极/消极/中性]
- 情绪依据：[简要说明]

3. 重要事件分析：
- 重大利好：[列出]
- 重大利空：[列出]
- 中性事件：[列出]

4. 潜在影响评估：
- 对公司基本面影响：[分析]
- 对股价短期影响：[分析]
- 对行业影响：[分析]

5. 风险提示：
- 主要风险点：[列出]
- 应对建议：[给出]
"""

async def get_stock_news_async(stock_code):
    """异步获取股票新闻"""
    loop = asyncio.get_event_loop()
    try:
        news_df = await loop.run_in_executor(thread_pool, ak.stock_news_em, stock_code)
        
        if not isinstance(news_df, pd.DataFrame):
            logger.error(f"获取到的新闻数据格式不正确: {type(news_df)}")
            return None
            
        required_columns = ['发布时间', '新闻标题', '新闻内容']
        missing_columns = [col for col in required_columns if col not in news_df.columns]
        if missing_columns:
            logger.error(f"新闻数据缺少必要列: {missing_columns}")
            return None
        
        news_df['发布时间'] = pd.to_datetime(news_df['发布时间'])
        seven_days_ago = datetime.datetime.now() - datetime.timedelta(days=7)
        news_df = news_df[news_df['发布时间'] >= seven_days_ago]
        
        return news_df if not news_df.empty else None
        
    except Exception as e:
        logger.error(f"获取股票{stock_code}新闻时出错: {str(e)}")
        return None

async def analyze_news_async(news_df):
    """异步分析新闻内容"""
    try:
        news_list = []
        valid_news_count = 0
        
        for _, row in news_df.head(5).iterrows():
            title = str(row.get('新闻标题', '无标题')).strip() or '无标题'
            content = str(row.get('新闻内容', '无内容')).strip() or '无内容'
            
            if title != '无标题' and content != '无内容':
                news_list.append(f"【新闻标题】：{title}\n【新闻内容】：{content}")
                valid_news_count += 1
                
        if valid_news_count == 0:
            return "最近7天的新闻均为空内容，建议查看其他新闻来源"
            
        news_content = "\n\n".join(news_list)
        prompt = NEWS_ANALYSIS_PROMPT.format(
            news_content=news_content,
            news_count=valid_news_count
        )
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            thread_pool,
            lambda: client.chat.completions.create(
                model=LLM_CONFIG["models"]["analysis"],
                messages=[{"role": "user", "content": prompt}]
            )
        )
        
        if not response or not response.choices:
            return "LLM返回结果为空"
            
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"分析新闻内容时出错: {str(e)}")
        return "处理新闻内容时出错"

async def get_news_sentiment_async(stock_code):
    """异步获取股票新闻并进行专业舆情分析"""
    try:
        news_df = await get_stock_news_async(stock_code)
        if news_df is None:
            return "无法获取新闻数据"
            
        analysis_result = await analyze_news_async(news_df)
        return analysis_result
        
    except Exception as e:
        logger.error(f"获取新闻舆情分析时出错: {str(e)}")
        return "无法获取新闻舆情分析"

@retry_on_exception(retries=3, delay=1)
def get_news_sentiment(stock_code):
    """同步版本的新闻舆情分析函数（为保持向后兼容）"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(get_news_sentiment_async(stock_code))
    finally:
        loop.close()

# 批量处理新闻舆情
async def process_news_sentiment_batch(stock_codes):
    """批量处理多个股票的新闻舆情"""
    tasks = [get_news_sentiment_async(code) for code in stock_codes]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return {
        code: result if not isinstance(result, Exception) else "处理出错"
        for code, result in zip(stock_codes, results)
    }
