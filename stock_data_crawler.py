import akshare as ak
import os
import pandas as pd
import threading
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_data_download.log'),
        logging.StreamHandler()
    ]
)

def download_stock_data(stock_code, stock_name, max_retries=3, retry_interval=5):
    for attempt in range(max_retries):
        try:
            # 获取历史行情数据
            stock_data = ak.stock_zh_a_hist(symbol=stock_code, period="daily", adjust="qfq")
            
            # 过滤近120天数据
            history_days_ago = pd.Timestamp.now() - pd.Timedelta(days=120)
            stock_data = stock_data[pd.to_datetime(stock_data['日期']) >= history_days_ago]
            
            # 保存为CSV文件
            filename = f"data/{stock_code}_{stock_name}.csv"
            stock_data.to_csv(filename, index=False)
            logging.debug(f"成功保存 {stock_name}({stock_code}) 的历史数据到 {filename}")
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"第 {attempt + 1} 次尝试获取 {stock_name}({stock_code}) 数据失败,{retry_interval}秒后重试... 错误信息: {str(e)}")
                time.sleep(retry_interval)
            else:
                logging.error(f"获取 {stock_name}({stock_code}) 数据失败,已达到最大重试次数 {max_retries}。错误信息: {str(e)}")
                return False

def main():
    # 创建data文件夹
    os.makedirs('data', exist_ok=True)

    # 获取A股市场市值前2000只股票列表并过滤ST股和停牌股票
    stock_list = ak.stock_zh_a_spot_em()
    
    # 获取ST股和停牌股票列表
    st_stocks = ak.stock_zh_a_st_em()['代码'].tolist()
    suspended_stocks = ak.stock_zh_a_stop_em()['代码'].tolist()
    
    # 过滤ST股和停牌股票
    filtered_stocks = stock_list[
        (~stock_list['代码'].isin(st_stocks)) & 
        (~stock_list['代码'].isin(suspended_stocks))
    ]
    
    top_stocks = filtered_stocks.sort_values(by='总市值', ascending=False).head(2000)

    # 使用线程池并发下载,添加进度条
    total_stocks = len(top_stocks)
    success_count = 0
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = []
        for index, row in top_stocks.iterrows():
            stock_code = row['代码']
            stock_name = row['名称']
            futures.append(executor.submit(download_stock_data, stock_code, stock_name))
        
        # 使用tqdm显示进度
        with tqdm(total=total_stocks, desc="下载进度") as pbar:
            for future in as_completed(futures):
                if future.result():
                    success_count += 1
                pbar.update(1)

    logging.info(f"所有股票数据抓取完成!成功下载 {success_count} 只股票数据,失败 {len(top_stocks) - success_count} 只")

if __name__ == "__main__":
    main()
