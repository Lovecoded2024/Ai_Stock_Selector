import pandas as pd
import os
import datetime
import logging
import random
import time
import asyncio
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from quantitative_analysis import StockAnalyzer
import model_processing
import sqlite3
from utils import retry_on_exception

# 配置重试参数
RETRY_COUNT = 3
RETRY_DELAY = 1
BATCH_SIZE = 250  # 批处理大小
MAX_CONCURRENT_REQUESTS = 100  # 最大并发请求数
API_RATE_LIMIT = 0.01  # 每个请求的最小间隔(秒)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def log_section(title):
    """添加日志分隔符"""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"{' ' * 10}{title.upper()}")
    logger.info(f"{'=' * 80}")

def get_all_stocks():
    """从数据库获取所有股票代码"""
    try:
        conn = sqlite3.connect('stock_data.db')
        cursor = conn.cursor()
        cursor.execute("SELECT stock_code FROM stock_info")
        all_codes = [row[0] for row in cursor.fetchall()]
        conn.close()
        return all_codes
    except Exception as e:
        logger.error(f"从数据库获取股票时出错: {str(e)}")
        raise

async def get_stock_data_async(stock_codes):
    """异步获取指定股票代码的数据"""
    analyzer = StockAnalyzer()
    data_frames = []
    
    try:
        # 初始化进度条
        analyzer.__class__.init_progress(len(stock_codes))
        
        # 连接数据库获取公司名称
        conn = sqlite3.connect('stock_data.db')
        cursor = conn.cursor()
        
        placeholders = ','.join('?' for _ in stock_codes)
        query = f"SELECT stock_code, COALESCE(stock_name, '未知公司') as stock_name FROM stock_info WHERE stock_code IN ({placeholders})"
        cursor.execute(query, stock_codes)
        company_names = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        
        # 使用线程池并行处理数据获取
        with ThreadPoolExecutor() as executor:
            loop = asyncio.get_event_loop()
            tasks = []
            for code in stock_codes:
                task = loop.run_in_executor(executor, analyzer.analyze_stock, code)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            for code, df in zip(stock_codes, results):
                if df is not None:
                    df['代码'] = code
                    df['公司名称'] = company_names.get(code, '未知公司')
                    data_frames.append(df)
        
    except Exception as e:
        logger.error(f"查询公司名称时出错: {str(e)}")
        raise
    
    if not data_frames:
        raise ValueError("未找到任何股票数据")
    
    combined = pd.concat(data_frames, ignore_index=True)
    combined['日期'] = pd.to_datetime(combined['日期'])
    combined = combined.sort_values(['代码', '日期'], ascending=[True, False])
    
    return combined.reset_index(drop=True)

class RateLimiter:
    """API请求限速器"""
    def __init__(self, rate_limit):
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.lock = asyncio.Lock()

    async def acquire(self):
        """获取请求许可"""
        async with self.lock:
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            if time_since_last_request < self.rate_limit:
                await asyncio.sleep(self.rate_limit - time_since_last_request)
            self.last_request_time = time.time()

async def process_single_stock(code, stock_df, rate_limiter):
    """处理单个股票"""
    if stock_df.empty:
        return None
    
    latest_data = stock_df.iloc[0].to_dict()
    try:
        await rate_limiter.acquire()
        rating_result = await model_processing.get_llm_rating_async(latest_data)
        return {
            '代码': code,
            '公司名称': stock_df.iloc[0]['公司名称'],
            '评分': rating_result['rating'],
            '分析': rating_result['analysis'],
            '建议': rating_result['recommendation'],
            'fundamental_analysis': rating_result.get('fundamental_analysis', '无基本面分析数据'),
            'trade_advice': rating_result.get('trade_advice', '无交易建议'),
            'risk_warning': rating_result.get('risk_warning', '无风险提示'),
            # 添加基础行情数据
            '最新价': latest_data.get('收盘', 0),
            '涨跌幅': latest_data.get('涨跌幅', 0),
            '换手率': latest_data.get('换手率', 0),
            '成交额': latest_data.get('成交额', 0),
            '成交量': latest_data.get('成交量', 0),
            # 添加量化指标数据
            'MA_5': latest_data.get('MA_5', 0),
            'MA_10': latest_data.get('MA_10', 0),
            'MA_20': latest_data.get('MA_20', 0),
            'MA_60': latest_data.get('MA_60', 0),
            'RSI_6': latest_data.get('RSI_6', 0),
            'RSI_12': latest_data.get('RSI_12', 0),
            'RSI_24': latest_data.get('RSI_24', 0),
            'MACD': latest_data.get('MACD', 0),
            'MACD_signal': latest_data.get('MACD_signal', 0),
            'MACD_hist': latest_data.get('MACD_hist', 0),
            'KDJ_K': latest_data.get('KDJ_K', 0),
            'KDJ_D': latest_data.get('KDJ_D', 0),
            'KDJ_J': latest_data.get('KDJ_J', 0),
            'BB_upper': latest_data.get('BB_upper', 0),
            'BB_middle': latest_data.get('BB_middle', 0),
            'BB_lower': latest_data.get('BB_lower', 0),
            'BB_width': latest_data.get('BB_width', 0),
            'ATR': latest_data.get('ATR', 0),
            'ADX': latest_data.get('ADX', 0),
            'VOLATILITY': latest_data.get('VOLATILITY', 0)
        }
    except Exception as e:
        logger.error(f"处理股票 {code} 时出错: {str(e)}")
        return {
            '代码': code,
            '公司名称': stock_df.iloc[0]['公司名称'],
            '评分': 0,
            '分析': '无法生成分析报告',
            '建议': '持有',
            'fundamental_analysis': '无法生成基本面分析',
            'trade_advice': '无法生成交易建议',
            'risk_warning': '无法生成风险提示',
            '最新价': 0,
            '涨跌幅': 0,
            '换手率': 0,
            '成交额': 0,
            '成交量': 0,
            # 添加量化指标默认值
            'MA_5': 0,
            'MA_10': 0,
            'MA_20': 0,
            'MA_60': 0,
            'RSI_6': 0,
            'RSI_12': 0,
            'RSI_24': 0,
            'MACD': 0,
            'MACD_signal': 0,
            'MACD_hist': 0,
            'KDJ_K': 0,
            'KDJ_D': 0,
            'KDJ_J': 0,
            'BB_upper': 0,
            'BB_middle': 0,
            'BB_lower': 0,
            'BB_width': 0,
            'ATR': 0,
            'ADX': 0,
            'VOLATILITY': 0
        }

async def process_stock_batch(stock_codes, stock_data):
    """异步批量处理股票"""
    rate_limiter = RateLimiter(API_RATE_LIMIT)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    async def process_with_semaphore(code):
        async with semaphore:
            stock_df = stock_data[stock_data['代码'] == code]
            return await process_single_stock(code, stock_df, rate_limiter)
    
    tasks = [process_with_semaphore(code) for code in stock_codes]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]

async def process_detailed_analysis_batch(stocks, stock_data):
    """异步批量处理股票的详细分析"""
    rate_limiter = RateLimiter(API_RATE_LIMIT)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    detailed_analyses = []
    
    async def process_single_analysis(stock):
        async with semaphore:
            code = stock['代码']
            stock_df = stock_data[stock_data['代码'] == code]
            if not stock_df.empty:
                try:
                    await rate_limiter.acquire()
                    return await model_processing.get_detailed_analysis_async(code, stock_df)
                except Exception as e:
                    logger.error(f"详细分析失败: {str(e)}")
                    return {
                        '代码': code,
                        '技术面分析': '无法生成技术面分析',
                        '基本面分析': '无法生成基本面分析',
                        '新闻舆情': '无法获取新闻舆情数据',
                        '交易建议': '无法生成交易建议',
                        '风险提示': '无法生成风险提示'
                    }
    
    tasks = [process_single_analysis(stock) for stock in stocks]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]

@retry_on_exception(retries=RETRY_COUNT, delay=RETRY_DELAY)
def generate_report(ratings, detailed_analyses):
    """生成分析报告"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""
╔══════════════════════════════════════════════════════════════════════════════
║                        股票筛选报告 - 生成时间:{timestamp}
╚══════════════════════════════════════════════════════════════════════════════\n"""
    
    detailed_dict = {analysis['代码']: analysis for analysis in detailed_analyses if isinstance(analysis, dict) and '代码' in analysis}
    sorted_ratings = sorted(ratings, key=lambda x: x['评分'], reverse=True)
    
    for i, stock in enumerate(sorted_ratings, 1):
        code = stock['代码']
        company_name = stock.get('公司名称', '未知公司')
        report += f"""
╔══════════════════════════════════════════════════════════════════════════════
║ 【{i}】股票代码:{code} - {company_name}
╚══════════════════════════════════════════════════════════════════════════════

▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 【基础分析】 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

评    分:{stock['评分']}/100
建    议:{stock['建议']}

【基础行情】
────────────────────────────────────────────────────────
最 新 价:{stock.get('最新价', 0):.2f}元
涨 跌 幅:{stock.get('涨跌幅', 0):.2f}%
换 手 率:{stock.get('换手率', 0):.2f}%
成 交 额:{stock.get('成交额', 0)/10000:.2f}万元
成 交 量:{stock.get('成交量', 0)/10000:.2f}万股

【量化指标】
────────────────────────────────────────────────────────
MA指标:
  MA5  :{stock.get('MA_5', 0):.2f}
  MA10 :{stock.get('MA_10', 0):.2f}
  MA20 :{stock.get('MA_20', 0):.2f}
  MA60 :{stock.get('MA_60', 0):.2f}

RSI指标:
  RSI6 :{stock.get('RSI_6', 0):.2f}
  RSI12:{stock.get('RSI_12', 0):.2f}
  RSI24:{stock.get('RSI_24', 0):.2f}

MACD指标:
  MACD     :{stock.get('MACD', 0):.3f}
  MACD信号 :{stock.get('MACD_signal', 0):.3f}
  MACD柱   :{stock.get('MACD_hist', 0):.3f}

KDJ指标:
  K值:{stock.get('KDJ_K', 0):.2f}
  D值:{stock.get('KDJ_D', 0):.2f}
  J值:{stock.get('KDJ_J', 0):.2f}

布林带:
  上轨:{stock.get('BB_upper', 0):.2f}
  中轨:{stock.get('BB_middle', 0):.2f}
  下轨:{stock.get('BB_lower', 0):.2f}
  带宽:{stock.get('BB_width', 0):.2f}%

其他指标:
  ATR    :{stock.get('ATR', 0):.3f}
  ADX    :{stock.get('ADX', 0):.2f}
  波动率 :{stock.get('VOLATILITY', 0):.2f}%

【1】技术面分析
────────────────────────────────────────────────────────
{stock.get('分析', '无分析数据')}

【2】基本面分析
────────────────────────────────────────────────────────
{stock.get('fundamental_analysis', '无基本面分析数据')}

【3】交易建议
────────────────────────────────────────────────────────
{stock.get('trade_advice', '无交易建议')}

【4】风险提示
────────────────────────────────────────────────────────
{stock.get('risk_warning', '无风险提示')}"""

        if code in detailed_dict:
            detailed = detailed_dict[code]
            report += f"""

▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 【深度分析】 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

【1】新闻舆情分析
────────────────────────────────────────────────────────
{detailed.get('新闻舆情', '无相关新闻舆情数据')}

【2】技术面分析(详细)
────────────────────────────────────────────────────────
{detailed.get('技术面分析', '无技术面分析数据')}

【3】基本面分析(详细)
────────────────────────────────────────────────────────
{detailed.get('基本面分析', '无基本面分析数据')}

【4】交易建议(详细)
────────────────────────────────────────────────────────
{detailed.get('交易建议', '无交易建议')}

【5】风险提示(详细)
────────────────────────────────────────────────────────
{detailed.get('风险提示', '无风险提示')}"""
        
        report += "\n"
    
    if not os.path.exists('report'):
        os.makedirs('report')
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
    report_path = f'report/stock_analysis_{timestamp}.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"报告已保存到 {report_path}")
    
    return report_path

async def main_async():
    start_time = time.time()
    try:
        # 0. 检查并更新数据库
        log_section("检查数据库")
        from db_maintenance import maintain_database
        maintain_database()
        
        # 1. 获取所有股票
        log_section("获取股票代码")
        all_codes = get_all_stocks()
        logger.info(f"共获取到{len(all_codes)}只股票")
        
        # 2. 获取股票数据并计算指标
        log_section("获取股票数据")
        stock_data = await get_stock_data_async(all_codes)
        logger.info("技术指标计算完成")
        
        # 3. 并行处理股票评分
        log_section("股票评分")
        all_ratings = []
        batches = [all_codes[i:i + BATCH_SIZE] for i in range(0, len(all_codes), BATCH_SIZE)]
        
        with tqdm(total=len(all_codes), desc="评分进度") as pbar:
            for batch in batches:
                batch_ratings = await process_stock_batch(batch, stock_data)
                all_ratings.extend(batch_ratings)
                pbar.update(len(batch))
        
        # 4. 选出评分最高的前10支股票进行深度分析
        log_section("深度分析")
        top_10_stocks = sorted(all_ratings, key=lambda x: x['评分'], reverse=True)[:10]
        
        # 5. 并行处理深度分析
        detailed_analyses = await process_detailed_analysis_batch(top_10_stocks, stock_data)
        
        # 6. 生成报告
        log_section("生成报告")
        report_path = generate_report(all_ratings, detailed_analyses)
        
        # 7. 输出结果摘要和性能统计
        end_time = time.time()
        execution_time = end_time - start_time
        avg_rating_time = execution_time / len(all_codes)
        avg_analysis_time = execution_time / len(top_10_stocks)
        
        print("\n╔═══════════════════════ 分析完成 ═══════════════════════╗")
        print(f"║ 完整报告已保存至: {report_path}")
        print(f"║ 总耗时: {execution_time:.2f} 秒")
        print(f"║ 平均评分耗时: {avg_rating_time:.2f} 秒/支")
        print(f"║ 平均分析耗时: {avg_analysis_time:.2f} 秒/支")
        print("║")
        print("║ 评分最高的10支股票:")
        for stock in top_10_stocks:
            print(f"║ 代码:{stock['代码']} - 评分:{stock['评分']} - 建议:{stock['建议']}")
        print("╚════════════════════════════════════════════════════════╝")
        
        logger.info(f"程序执行完成,总耗时: {execution_time:.2f} 秒,平均评分耗时: {avg_rating_time:.2f} 秒/支,平均分析耗时: {avg_analysis_time:.2f} 秒/支")
            
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        raise

def main():
    """主函数入口"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main_async())
    finally:
        loop.close()

if __name__ == "__main__":
    main()
