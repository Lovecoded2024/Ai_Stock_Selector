import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os
import logging
import traceback
from stock_data_crawler import download_stock_data
import akshare as ak
from concurrent.futures import ThreadPoolExecutor
import threading
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='db_maintenance.log'
)
logger = logging.getLogger(__name__)

def get_latest_date(conn, stock_code):
    """获取指定股票的最新交易日期"""
    try:
        cursor = conn.cursor()
        cursor.execute('''
        SELECT MAX(trade_date) FROM daily_quote 
        WHERE stock_code = ?
        ''', (stock_code,))
        result = cursor.fetchone()[0]
        return result if result else None
    except Exception as e:
        logger.error(f"获取股票 {stock_code} 最新日期失败: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def validate_row_data(row):
    """验证单行数据是否有效"""
    try:
        # 检查关键字段是否存在且有效
        required_fields = ['开盘', '收盘', '最高', '最低', '成交量', '成交额']
        for field in required_fields:
            if pd.isna(row[field]) or row[field] == '':
                return False
        return True
    except Exception:
        return False

def check_and_update_stock(csv_path):
    """检查并更新单个股票的数据"""
    try:
        filename = os.path.basename(csv_path)
        stock_code = filename.split('_')[0]
        
        # 创建线程独立的数据库连接
        thread_conn = sqlite3.connect('stock_data.db')
        
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        df['日期'] = pd.to_datetime(df['日期'])
        
        # 获取数据库中的最新日期
        latest_date = get_latest_date(thread_conn, stock_code)
        if latest_date:
            latest_date = datetime.strptime(latest_date, '%Y-%m-%d')
            
            # 筛选出新数据
            new_data = df[df['日期'] > latest_date]
            if not new_data.empty:
                # 过滤无效数据
                valid_data = new_data[new_data.apply(validate_row_data, axis=1)]
                invalid_count = len(new_data) - len(valid_data)
                if invalid_count > 0:
                    logger.warning(f"股票 {stock_code} 发现 {invalid_count} 条无效数据")
                
                if not valid_data.empty:
                    # 准备批量插入数据
                    data_to_insert = []
                    for _, row in valid_data.iterrows():
                        data_to_insert.append((
                            stock_code,
                            row['日期'].strftime('%Y-%m-%d'),
                            row['开盘'],
                            row['收盘'],
                            row['最高'],
                            row['最低'],
                            row['成交量'],
                            row['成交额'],
                            row['振幅'],
                            row['涨跌幅'],
                            row['涨跌额'],
                            row['换手率']
                        ))
                    
                    # 批量插入
                    cursor = thread_conn.cursor()
                    cursor.executemany('''
                    INSERT OR REPLACE INTO daily_quote 
                    (stock_code, trade_date, open_price, close_price, high_price, low_price, 
                     volume, amount, amplitude, change_percent, change_amount, turnover_rate)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', data_to_insert)
                    thread_conn.commit()
                    logger.info(f"股票 {stock_code} 更新了 {len(valid_data)} 条新数据")
        else:
            # 如果数据库中没有该股票数据,执行完整导入
            # 过滤无效数据
            valid_data = df[df.apply(validate_row_data, axis=1)]
            invalid_count = len(df) - len(valid_data)
            if invalid_count > 0:
                logger.warning(f"股票 {stock_code} 发现 {invalid_count} 条无效数据")
            
            if not valid_data.empty:
                # 准备批量插入数据
                data_to_insert = []
                for _, row in valid_data.iterrows():
                    data_to_insert.append((
                        stock_code,
                        row['日期'].strftime('%Y-%m-%d'),
                        row['开盘'],
                        row['收盘'],
                        row['最高'],
                        row['最低'],
                        row['成交量'],
                        row['成交额'],
                        row['振幅'],
                        row['涨跌幅'],
                        row['涨跌额'],
                        row['换手率']
                    ))
                
                # 批量插入
                cursor = thread_conn.cursor()
                cursor.executemany('''
                INSERT OR REPLACE INTO daily_quote 
                (stock_code, trade_date, open_price, close_price, high_price, low_price, 
                 volume, amount, amplitude, change_percent, change_amount, turnover_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', data_to_insert)
                thread_conn.commit()
                logger.info(f"股票 {stock_code} 完成首次数据导入,插入了 {len(valid_data)} 条有效数据")
            else:
                logger.error(f"股票 {stock_code} 没有有效数据可供导入")
                
    except Exception as e:
        logger.error(f"处理股票 {stock_code} 数据失败: {str(e)}")
        logger.error(traceback.format_exc())

def get_all_stocks_latest_date(conn):
    """获取数据库中所有股票的最新交易日期"""
    try:
        cursor = conn.cursor()
        cursor.execute('''
        SELECT stock_code, MAX(trade_date) as latest_date 
        FROM daily_quote 
        GROUP BY stock_code
        ''')
        return dict(cursor.fetchall())
    except Exception as e:
        logger.error(f"获取所有股票最新日期失败: {str(e)}")
        logger.error(traceback.format_exc())
        return {}

def get_stock_name(stock_code):
    """从akshare获取股票名称"""
    try:
        stock_info = ak.stock_zh_a_spot_em()
        stock_info = stock_info[stock_info['代码'] == stock_code]
        if not stock_info.empty:
            return stock_info.iloc[0]['名称']
    except:
        pass
    return None

def get_realtime_quote(stock_code):
    """获取单个股票的实时行情"""
    try:
        # 使用akshare获取实时行情
        realtime_data = ak.stock_zh_a_spot_em()
        stock_data = realtime_data[realtime_data['代码'] == stock_code]
        
        if not stock_data.empty:
            data = stock_data.iloc[0]
            
            # 检查必要字段是否存在且有效
            required_fields = {
                '今开': 'open_price',
                '最新价': 'close_price',
                '最高': 'high_price',
                '最低': 'low_price',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'change_percent',
                '涨跌额': 'change_amount',
                '换手率': 'turnover_rate'
            }
            
            result = {
                'stock_code': stock_code,
                'trade_date': datetime.now().strftime('%Y-%m-%d')
            }
            
            # 检查每个字段并转换
            for cn_field, en_field in required_fields.items():
                if cn_field not in data or pd.isna(data[cn_field]):
                    logger.warning(f"股票 {stock_code} 缺少字段 {cn_field}")
                    return None
                try:
                    result[en_field] = float(data[cn_field])
                except (ValueError, TypeError):
                    logger.warning(f"股票 {stock_code} 字段 {cn_field} 数据无效: {data[cn_field]}")
                    return None
            
            return result
            
        logger.warning(f"未找到股票 {stock_code} 的实时数据")
        return None
    except Exception as e:
        logger.error(f"获取股票 {stock_code} 实时行情失败: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def update_realtime_quote(conn, quote_data):
    """更新单个股票的实时行情到数据库"""
    try:
        cursor = conn.cursor()
        cursor.execute('''
        INSERT OR REPLACE INTO daily_quote 
        (stock_code, trade_date, open_price, close_price, high_price, low_price, 
         volume, amount, amplitude, change_percent, change_amount, turnover_rate)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            quote_data['stock_code'],
            quote_data['trade_date'],
            quote_data['open_price'],
            quote_data['close_price'],
            quote_data['high_price'],
            quote_data['low_price'],
            quote_data['volume'],
            quote_data['amount'],
            quote_data['amplitude'],
            quote_data['change_percent'],
            quote_data['change_amount'],
            quote_data['turnover_rate']
        ))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"更新股票 {quote_data['stock_code']} 实时行情失败: {str(e)}")
        return False

def process_stock(stock_code, stock_name):
    """处理单个股票的数据更新"""
    try:
        # 先获取实时行情并更新
        realtime_quote = get_realtime_quote(stock_code)
        if realtime_quote:
            thread_conn = sqlite3.connect('stock_data.db')
            update_realtime_quote(thread_conn, realtime_quote)
            thread_conn.close()
        
        # 然后更新历史数据
        if download_stock_data(stock_code, stock_name):
            # 如果下载成功,更新数据库
            csv_path = os.path.join('data', f"{stock_code}_{stock_name}.csv")
            if os.path.exists(csv_path):
                check_and_update_stock(csv_path)
            else:
                logger.error(f"未找到股票 {stock_code} 的CSV文件")
        else:
            logger.error(f"下载股票 {stock_code} 数据失败")
    except Exception as e:
        logger.error(f"处理股票 {stock_code} 失败: {str(e)}")

def maintain_database():
    """维护数据库,检查并更新所有股票数据"""
    try:
        conn = sqlite3.connect('stock_data.db')
        
        # 获取所有股票的最新日期
        stocks_latest_date = get_all_stocks_latest_date(conn)
        conn.close()
        today = datetime.now().date()
        
        # 检查每只股票的数据是否最新
        outdated_stocks = []
        for stock_code, latest_date in stocks_latest_date.items():
            if latest_date:
                latest_date = datetime.strptime(latest_date, '%Y-%m-%d').date()
                # 如果最新数据日期与当前日期差距超过3个工作日,认为需要更新
                if (today - latest_date).days > 3:
                    outdated_stocks.append(stock_code)
            else:
                outdated_stocks.append(stock_code)
        
        if outdated_stocks:
            print(f"开始更新 {len(outdated_stocks)} 只股票数据...")
            
            # 使用线程池并行处理过期的股票数据
            with ThreadPoolExecutor(max_workers=50) as executor:
                futures = []
                # 创建进度条
                pbar = tqdm(total=len(outdated_stocks), desc="更新进度")
                
                for stock_code in outdated_stocks:
                    stock_name = get_stock_name(stock_code)
                    if stock_name:
                        future = executor.submit(process_stock, stock_code, stock_name)
                        future.add_done_callback(lambda p: pbar.update(1))
                        futures.append(future)
                    else:
                        logger.error(f"获取股票 {stock_code} 名称失败")
                        pbar.update(1)
                
                # 等待所有任务完成
                for future in futures:
                    future.result()
                
                pbar.close()
            print("数据库更新完成!")
        else:
            logger.info("所有股票数据均为最新")
    except Exception as e:
        logger.error(f"数据库维护失败: {str(e)}")
        logger.error(traceback.format_exc())

def update_realtime_quotes():
    """更新所有股票的实时行情"""
    try:
        conn = sqlite3.connect('stock_data.db')
        
        # 获取所有股票代码
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT stock_code FROM daily_quote')
        stock_codes = [row[0] for row in cursor.fetchall()]
        
        print(f"开始更新 {len(stock_codes)} 只股票实时行情...")
        
        # 使用线程池并行获取实时行情
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = []
            success_count = 0
            
            # 创建进度条
            pbar = tqdm(total=len(stock_codes), desc="更新实时行情")
            
            for stock_code in stock_codes:
                future = executor.submit(get_realtime_quote, stock_code)
                future.add_done_callback(lambda p: pbar.update(1))
                futures.append(future)
            
            # 处理结果
            for future in futures:
                quote_data = future.result()
                if quote_data and update_realtime_quote(conn, quote_data):
                    success_count += 1
            
            pbar.close()
        
        conn.close()
        print(f"实时行情更新完成! 成功更新 {success_count} 只股票")
        
    except Exception as e:
        logger.error(f"更新实时行情失败: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == '__main__':
    maintain_database()
    update_realtime_quotes()
