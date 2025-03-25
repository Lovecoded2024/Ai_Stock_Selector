import os
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from utils import retry_on_exception

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入配置
from config import LLM_CONFIG, MODEL_CONFIG, THREAD_POOL_CONFIG

# 配置LLM
client = OpenAI(
    base_url=LLM_CONFIG["base_url"],
    api_key=LLM_CONFIG["api_key"]
)

# 创建线程池
thread_pool = ThreadPoolExecutor(max_workers=THREAD_POOL_CONFIG["max_workers"])

def format_stock_data(stock_data):
    """格式化股票数据,包含更完整的量化指标"""
    try:
        current_price = float(stock_data['收盘'])
        prev_price = float(stock_data['开盘'])
        high_price = float(stock_data['最高'])
        low_price = float(stock_data['最低'])
    except (ValueError, TypeError):
        current_price = prev_price = high_price = low_price = 0.0

    def safe_format(value):
        try:
            return f"{float(value):.2f}"
        except (ValueError, TypeError):
            return "N/A"

    # 计算价格变动百分比
    price_change_pct = ((current_price - prev_price) / prev_price * 100) if prev_price != 0 else 0

    return {
        # 价格数据
        'current_price': current_price,
        'prev_price': prev_price,
        'high_price': high_price,
        'low_price': low_price,
        'price_change_pct': f"{price_change_pct:.2f}",
        'volume': safe_format(stock_data['成交量']),
        'amount': safe_format(stock_data['成交额']),
        
        # 趋势指标
        'MA_5': safe_format(stock_data['MA_5']),
        'MA_10': safe_format(stock_data['MA_10']),
        'MA_20': safe_format(stock_data['MA_20']),
        'MA_30': safe_format(stock_data['MA_30']),
        'MA_60': safe_format(stock_data['MA_60']),
        'EMA_12': safe_format(stock_data['EMA_12']),
        'EMA_26': safe_format(stock_data['EMA_26']),
        
        # 动量指标
        'RSI_6': safe_format(stock_data['RSI_6']),
        'RSI_12': safe_format(stock_data['RSI_12']),
        'RSI_24': safe_format(stock_data['RSI_24']),
        'MACD': safe_format(stock_data['MACD']),
        'MACD_signal': safe_format(stock_data['MACD_signal']),
        'MACD_hist': safe_format(stock_data['MACD_hist']),
        
        # 波动指标
        'BB_upper': safe_format(stock_data['BB_upper']),
        'BB_middle': safe_format(stock_data['BB_middle']),
        'BB_lower': safe_format(stock_data['BB_lower']),
        'BB_width': safe_format(stock_data['BB_width']),
        'ATR': safe_format(stock_data['ATR']),
        
        # 动能指标
        'KDJ_K': safe_format(stock_data['KDJ_K']),
        'KDJ_D': safe_format(stock_data['KDJ_D']),
        'KDJ_J': safe_format(stock_data['KDJ_J']),
        'MOM_10': safe_format(stock_data['MOM_10']),
        'MOM_20': safe_format(stock_data['MOM_20']),
        
        # 成交量指标
        'WILLR': safe_format(stock_data['WILLR']),
        'OBV': safe_format(stock_data['OBV']),
        'VOLUME_MA5': safe_format(stock_data['VOLUME_MA5']),
        'VOLUME_MA10': safe_format(stock_data['VOLUME_MA10']),
        
        # 波动率指标
        'VOLATILITY': safe_format(stock_data['VOLATILITY']),
        'VOLATILITY_MA': safe_format(stock_data['VOLATILITY_MA'])
    }

async def get_llm_rating_async(stock_data, max_retries=3, retry_delay=2):
    """异步获取LLM评分
    
    Args:
        stock_data: 股票数据
        max_retries: 最大重试次数
        retry_delay: 重试延迟(秒)
    
    Returns:
        dict: 包含评分和分析结果的字典
    
    Raises:
        ValueError: 数据格式错误
        Exception: API调用失败或其他错误
    """
    try:
        formatted_data = format_stock_data(stock_data)
    except Exception as e:
        logger.error(f"数据格式化失败: {str(e)}")
        raise ValueError(f"股票数据格式错误: {str(e)}")
    
    prompt = f"""
    你是一位专业的股票分析师。请基于以下详细的技术指标数据进行分析,并严格按照评分标准和格式返回结果。

    价格与成交信息:
    当前价格:{formatted_data['current_price']:.2f}元
    开盘价格:{formatted_data['prev_price']:.2f}元
    最高/最低:{formatted_data['high_price']:.2f}/{formatted_data['low_price']:.2f}元
    价格变动:{formatted_data['price_change_pct']}%
    成交量:{formatted_data['volume']}
    成交额:{formatted_data['amount']}

    趋势指标:
    - 移动平均线(MA):
      5日均线:{formatted_data['MA_5']}
      10日均线:{formatted_data['MA_10']}
      20日均线:{formatted_data['MA_20']}
      30日均线:{formatted_data['MA_30']}
      60日均线:{formatted_data['MA_60']}
    - 指数移动平均线(EMA):
      12日EMA:{formatted_data['EMA_12']}
      26日EMA:{formatted_data['EMA_26']}

    动量指标:
    - RSI指标:
      6日RSI:{formatted_data['RSI_6']}
      12日RSI:{formatted_data['RSI_12']}
      24日RSI:{formatted_data['RSI_24']}
    - MACD指标:
      MACD:{formatted_data['MACD']}
      信号线:{formatted_data['MACD_signal']}
      柱状图:{formatted_data['MACD_hist']}

    波动指标:
    - 布林带:
      上轨:{formatted_data['BB_upper']}
      中轨:{formatted_data['BB_middle']}
      下轨:{formatted_data['BB_lower']}
      带宽:{formatted_data['BB_width']}
    - ATR:{formatted_data['ATR']}

    动能指标:
    - KDJ指标:
      K值:{formatted_data['KDJ_K']}
      D值:{formatted_data['KDJ_D']}
      J值:{formatted_data['KDJ_J']}
    - 动量指标:
      10日动量:{formatted_data['MOM_10']}
      20日动量:{formatted_data['MOM_20']}

    成交量指标:
    - 威廉指标:{formatted_data['WILLR']}
    - OBV能量潮:{formatted_data['OBV']}
    - 成交量均线:
      5日均量:{formatted_data['VOLUME_MA5']}
      10日均量:{formatted_data['VOLUME_MA10']}

    波动率指标:
    - 当前波动率:{formatted_data['VOLATILITY']}
    - 波动率均值:{formatted_data['VOLATILITY_MA']}
    
    请基于以下量化标准进行综合评分(0-100分),并按格式返回分析结果:

    评分维度及权重:
    [趋势确定性 45%]
    - 均线系统(20%):
      * 5/10/20/30/60日均线多头排列(20分)
      * 部分均线多头排列(10分)
      * 空头排列(-10分)
      * 金叉信号额外+5分,死叉信号-5分
    - MACD趋势(15%):
      * 柱状图连续3日扩张(每日+3分)
      * 柱状图连续3日缩量(每日-2分)
      * MACD金叉+5分,死叉-5分
    - 量价配合(10%):
      * 上涨成交量>5日均量(+2分/日)
      * 下跌成交量<5日均量(+1分/日)
      * 上涨缩量或下跌放量(-2分/日)
    
    [买入时机 40%]
    - 布林带位置(15%):
      * 价格接近下轨(±2%内)(+15分)
      * 价格在中轨附近(±3%内)(+8分)
      * 价格接近上轨(±2%内)(-5分)
    - RSI状态(10%):
      * RSI6 < 20 (+10分)
      * RSI6 20-30 (+5分)
      * RSI6 30-70 (0分)
      * RSI6 70-80 (-5分)
      * RSI6 > 80 (-10分)
    - 支撑强度(15%):
      * 3条以上均线支撑(+15分)
      * 2条均线支撑(+10分)
      * 1条均线支撑(+5分)
      * 无均线支撑(-5分)
    
    [风险系数 15%]
    - 波动率评估(8%):
      * ATR < 20日均值(-2分)
      * ATR > 20日均值+20%(-8分)
      * 其他情况(0分)
    - 止损空间(7%):
      * 距支撑位<3%(+7分)
      * 距支撑位3-5%(+5分)
      * 距支撑位5-8%(+3分)
      * 距支撑位>8%(0分)
    
    动态调整项(±5分):
    - 多重信号共振:
      * MACD金叉 + RSI超卖 + 放量突破(+5分)
      * MACD死叉 + RSI超买 + 缩量(-5分)
    - 背离信号:
      * 顶背离:价格创新高但指标未创新高(-3分/个)
      * 底背离:价格创新低但指标未创新低(+3分/个)

    投资建议阈值:
    - 评分 >= 75分:建议"买入",可考虑30-50%仓位
    - 评分 60-74分:建议"持有",可考虑15-30%仓位
    - 评分 < 60分:建议"卖出",可考虑清仓或降低仓位至15%以下

    请严格按照以下格式返回分析结果,每部分占一行:
    第1行:仅包含0-100之间的数字评分,不含其他文字
    第2行:趋势确定性分析(包括:趋势方向、持续性、指标一致性、量价配合)
    第3行:买入时机分析(包括:价格位置、技术指标、支撑位、量能特征)
    第4行:详细的交易建议(包括:建议仓位、买入区间、止损位、目标位)
    第5行:风险提示(包括:趋势转折风险、技术指标风险、流动性风险)
    第6行:仅包含"买入"、"持有"或"卖出"三个词之一
    """
    
    for attempt in range(max_retries):
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                thread_pool,
                lambda: client.chat.completions.create(
                    model=MODEL_CONFIG["rating_model"],
                    messages=[{"role": "user", "content": prompt}],
                    timeout=30  # 设置超时时间
                )
            )
        
            content = response.choices[0].message.content.strip()
            parts = content.split('\n', 5)
            
            if len(parts) != 6:
                raise ValueError("模型返回格式错误: 需要6行数据")
            
            try:
                rating = float(parts[0].strip())
                if not (0 <= rating <= 100):
                    raise ValueError("评分必须在0-100之间")
            except ValueError as e:
                raise ValueError(f"评分格式错误: {str(e)}")
            
            tech_analysis = parts[1].strip()
            if not tech_analysis:
                raise ValueError("技术面分析为空")
                
            fundamental_analysis = parts[2].strip()
            if not fundamental_analysis:
                raise ValueError("基本面分析为空")
                
            trade_advice = parts[3].strip()
            if not trade_advice:
                raise ValueError("交易建议为空")
                
            risk_warning = parts[4].strip()
            if not risk_warning:
                raise ValueError("风险提示为空")
                
            # 根据评分强制执行建议
            if rating >= 75:
                recommendation = '买入'
            elif rating >= 60:
                recommendation = '持有'
            else:
                recommendation = '卖出'
                
            return {
                'rating': rating,
                'analysis': tech_analysis,
                'fundamental_analysis': fundamental_analysis,
                'trade_advice': trade_advice,
                'risk_warning': risk_warning,
                'recommendation': recommendation,  # 使用根据评分计算的建议
                'retry_count': attempt
            }
            
        except (ValueError, KeyError) as e:
            # 数据格式错误,直接抛出
            logger.error(f"数据验证失败: {str(e)}")
            if attempt == max_retries - 1:
                raise
                
        except Exception as e:
            # 其他错误(如网络错误)尝试重试
            logger.warning(f"第{attempt + 1}次调用失败: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                continue
            else:
                logger.error(f"达到最大重试次数{max_retries},最后一次错误: {str(e)}")
                raise
                
    # 如果所有重试都失败,返回默认值
    logger.error("所有重试均失败,返回默认值")
    return {
        'rating': 50,
        'analysis': '无法生成技术面分析',
        'fundamental_analysis': '无法生成基本面分析',
        'trade_advice': '无法生成交易建议',
        'risk_warning': '无法生成风险提示',
        'recommendation': '持有',
        'retry_count': max_retries
    }

async def get_detailed_analysis_async(stock_code, stock_data, max_retries=3, retry_delay=2):
    """异步获取详细分析
    
    Args:
        stock_code: 股票代码
        stock_data: 股票数据
        max_retries: 最大重试次数
        retry_delay: 重试延迟(秒)
    
    Returns:
        dict: 包含详细分析结果的字典
    
    Raises:
        ValueError: 数据格式错误
        Exception: API调用失败或其他错误
    """
    from news_sentiment import get_news_sentiment_async
    
    if not stock_code:
        raise ValueError("股票代码不能为空")
        
    if stock_data is None or stock_data.empty:
        raise ValueError("股票数据不能为空")
    
    # 格式化股票数据
    stock_data = stock_data.sort_values('日期', ascending=False)
    prices = stock_data['收盘'].astype(float).values
    
    latest_price = stock_data.iloc[0]['收盘'] if not stock_data.empty else 0.0
    price_range = {
        'high': prices.max() if len(prices) > 0 else 0.0,
        'low': prices.min() if len(prices) > 0 else 0.0
    }
    
    # 格式化数据显示
    # 格式化历史数据
    data_str = "最近30天技术指标趋势：\n"
    for _, row in stock_data.head(30).iterrows():
        data_str += (
            f"日期: {row['日期']}\n"
            f"价格: 开盘{row['开盘']}/收盘{row['收盘']}/最高{row['最高']}/最低{row['最低']}\n"
            f"成交: 量{row['成交量']}/额{row['成交额']}\n"
            f"均线: MA5={row['MA_5']:.2f}/MA10={row['MA_10']:.2f}/MA20={row['MA_20']:.2f}/MA60={row['MA_60']:.2f}\n"
            f"MACD: DIF={row['MACD']:.2f}/DEA={row['MACD_signal']:.2f}/HIST={row['MACD_hist']:.2f}\n"
            f"KDJ: K={row['KDJ_K']:.2f}/D={row['KDJ_D']:.2f}/J={row['KDJ_J']:.2f}\n"
            f"RSI: 6={row['RSI_6']:.2f}/12={row['RSI_12']:.2f}/24={row['RSI_24']:.2f}\n"
            f"布林带: 上={row['BB_upper']:.2f}/中={row['BB_middle']:.2f}/下={row['BB_lower']:.2f}/宽={row['BB_width']:.2f}\n"
            f"波动性: ATR={row['ATR']:.2f}/VOL={row['VOLATILITY']:.2f}\n"
            "---\n"
        )
    
    full_prompt = f"""
    请对股票{stock_code}进行深度分析。

    {data_str}

    请严格按照以下格式返回，必须使用单行文本，用"|"分隔五个部分：

    [技术面分析]|[基本面分析]|[新闻舆情分析]|[交易建议]|[风险提示]

    注意事项：
    1. 返回必须是单行文本，不能包含任何换行符
    2. 五个部分必须按照上述顺序，使用"|"分隔
    3. 每个部分的内容不能包含"|"字符
    4. 所有价格必须使用实际数值，精确到分
    5. 不要添加任何额外的标签或分隔符

    各部分内容要求：
    [技术面分析] 必须包含：价格趋势分析；支撑位和阻力位分析（基于技术指标自主判断）；趋势和成交量配合度；主要技术指标综合研判。

    [基本面分析] 必须包含：所属行业分析；市场地位评估；相对估值分析。

    [新闻舆情分析] 必须包含：近期重要新闻概述；市场情绪评估；潜在影响分析。

    [交易建议] 必须包含：趋势研判；建议仓位；买入区间（基于技术指标自主判断）；止损位和目标位；建议持仓周期。

    [风险提示] 必须包含：技术面风险；基本面风险；市场风险；具体控制建议。

    示例格式：
    当前价格分析显示...|从行业角度来看...|近期新闻显示...|建议以30%仓位在...|主要风险包括...
    """
    
    for attempt in range(max_retries):
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                thread_pool,
                lambda: client.chat.completions.create(
                    model=MODEL_CONFIG["analysis_model"],
                    messages=[
                        {"role": "system", "content": "你是一位专业的股票分析师,擅长技术分析和基本面分析。请严格按照要求的格式提供分析结果。"},
                        {"role": "user", "content": full_prompt}
                    ],
                    timeout=30  # 设置超时时间
                )
            )
        
            parts = response.choices[0].message.content.split('|')
            if len(parts) != 5:
                raise ValueError("模型返回格式错误: 需要5个部分")
                
            # 验证每个部分都不为空
            for i, part in enumerate(parts):
                if not part.strip():
                    raise ValueError(f"第{i+1}部分分析内容为空")
            
            try:
                # 获取新闻舆情分析,如果失败则使用模型生成的备选内容
                news_content = await get_news_sentiment_async(stock_code)
                if not news_content or news_content == "无法获取新闻舆情数据":
                    news_content = parts[2].strip()
                    logger.warning(f"使用模型生成的舆情分析作为备选: {stock_code}")
            except Exception as e:
                logger.error(f"获取新闻舆情失败: {str(e)}")
                news_content = parts[2].strip()
            
            result = {
                '代码': stock_code,
                '技术面分析': parts[0].strip(),
                '基本面分析': parts[1].strip(),
                '新闻舆情': news_content,
                '交易建议': parts[3].strip(),
                '风险提示': parts[4].strip(),
                'retry_count': attempt
            }
            
            # 验证所有必要字段
            for key, value in result.items():
                if not value and key != 'retry_count':
                    raise ValueError(f"字段{key}不能为空")
                    
            return result
            
        except (ValueError, KeyError) as e:
            # 数据格式错误,直接抛出
            logger.error(f"数据验证失败: {str(e)}")
            if attempt == max_retries - 1:
                raise
                
        except Exception as e:
            # 其他错误(如网络错误)尝试重试
            logger.warning(f"第{attempt + 1}次调用失败: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                continue
            else:
                logger.error(f"达到最大重试次数{max_retries},最后一次错误: {str(e)}")
                raise
                
    # 如果所有重试都失败,返回默认值
    logger.error("所有重试均失败,返回默认值")
    return {
        '代码': stock_code,
        '技术面分析': '无法生成技术面分析',
        '基本面分析': '无法生成基本面分析',
        '新闻舆情': '无法获取新闻舆情数据',
        '交易建议': '无法生成交易建议',
        '风险提示': '无法生成风险提示',
        'retry_count': max_retries
    }

# 为了保持向后兼容，保留同步版本的函数
@retry_on_exception(retries=3, delay=1)
def get_llm_rating(stock_data):
    """同步版本的LLM评分函数"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(get_llm_rating_async(stock_data))
    finally:
        loop.close()

@retry_on_exception(retries=3, delay=1)
def get_detailed_analysis(stock_code, stock_data):
    """同步版本的详细分析函数"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(get_detailed_analysis_async(stock_code, stock_data))
    finally:
        loop.close()
