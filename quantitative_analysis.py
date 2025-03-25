import pandas as pd
import numpy as np
import talib
import sqlite3
import logging
from datetime import datetime
from tqdm import tqdm
from db_maintenance import maintain_database

class StockAnalyzer:
    _pbar = None
    _total_stocks = 0
    _processed_stocks = 0
    
    def __init__(self, db_path='stock_data.db'):
        self.db_path = db_path
        
    @classmethod
    def init_progress(cls, total):
        """初始化进度条"""
        cls._total_stocks = total
        cls._processed_stocks = 0
        cls._pbar = tqdm(total=total, desc="获取股票数据")
    
    @classmethod
    def update_progress(cls):
        """更新进度条"""
        if cls._pbar is not None:
            cls._processed_stocks += 1
            cls._pbar.update(1)
            if cls._processed_stocks >= cls._total_stocks:
                cls._pbar.close()
                cls._pbar = None
        
    def get_connection(self):
        """获取数据库连接"""
        return sqlite3.connect(self.db_path)
    
    def check_data_freshness(self):
        """检查数据是否是最新的"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT MAX(trade_date) FROM daily_quote')
        latest_date = cursor.fetchone()[0]
        
        if not latest_date:
            conn.close()
            return False
            
        latest_date = datetime.strptime(latest_date, '%Y-%m-%d')
        current_date = datetime.now()
        
        if (current_date - latest_date).days > 3:
            conn.close()
            return False
            
        conn.close()
        return True
    
    def update_if_needed(self):
        """如果需要则更新数据"""
        if not self.check_data_freshness():
            logging.info("数据不是最新的,正在更新...")
            maintain_database()
            return True
        logging.debug("数据已是最新")
        return False
    
    def get_stock_data(self, stock_code, start_date=None, end_date=None):
        """获取指定股票的数据"""
        conn = self.get_connection()
        
        query = '''
        SELECT trade_date as 日期, 
               open_price as 开盘, 
               close_price as 收盘,
               high_price as 最高,
               low_price as 最低,
               volume as 成交量,
               amount as 成交额,
               turnover_rate as 换手率,
               change_percent as 涨跌幅
        FROM daily_quote 
        WHERE stock_code = ?
        '''
        params = [stock_code]
        
        if start_date:
            query += ' AND trade_date >= ?'
            params.append(start_date)
        if end_date:
            query += ' AND trade_date <= ?'
            params.append(end_date)
            
        query += ' ORDER BY trade_date'
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    def calculate_indicators(self, df):
        """计算技术指标"""
        # 确保数据按日期排序
        df = df.sort_values('日期')
        
        close_prices = df['收盘'].values
        volumes = df['成交量'].values
        
        # 移动平均线
        for period in [5, 10, 20, 30, 60]:
            # 价格移动平均线
            ma_series = pd.Series(talib.MA(close_prices, timeperiod=period))
            df[f'MA_{period}'] = ma_series.bfill()
            
            # 成交量移动平均线
            volume_ma = pd.Series(talib.MA(volumes, timeperiod=period))
            df[f'VOLUME_MA{period}'] = volume_ma.bfill()
        
        # 指数移动平均线 - 添加更多周期
        for period in [12, 26]:  # 添加MACD常用的周期
            ema_series = pd.Series(talib.EMA(close_prices, timeperiod=period))
            df[f'EMA_{period}'] = ema_series.bfill()
        
        # 相对强弱指数
        for period in [6, 12, 24]:
            df[f'RSI_{period}'] = talib.RSI(close_prices, timeperiod=period)
        
        # 布林带
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(
            close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        # 计算布林带宽度
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle'] * 100
        
        # MACD
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(
            close_prices, fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # KDJ随机指标
        df['KDJ_K'], df['KDJ_D'] = talib.STOCH(
            df['最高'].values, df['最低'].values, close_prices,
            fastk_period=9, slowk_period=3, slowk_matype=0,
            slowd_period=3, slowd_matype=0
        )
        df['KDJ_J'] = 3 * df['KDJ_K'] - 2 * df['KDJ_D']
        
        # 平均真实波幅
        df['ATR'] = talib.ATR(df['最高'].values, df['最低'].values, close_prices, timeperiod=14)
        
        # 动量指标
        for period in [10, 20]:
            df[f'MOM_{period}'] = talib.MOM(close_prices, timeperiod=period)
        
        # 威廉指标
        df['WILLR'] = talib.WILLR(df['最高'].values, df['最低'].values, close_prices, timeperiod=14)
        
        # 能量潮指标
        df['OBV'] = talib.OBV(close_prices, volumes)
        
        # 波动率相关指标
        df['VOLATILITY'] = df['ATR'] / df['收盘'] * 100
        # 波动率移动平均
        df['VOLATILITY_MA'] = df['VOLATILITY'].rolling(window=20).mean()
        
        # 趋势强度指标
        df['ADX'] = talib.ADX(df['最高'].values, df['最低'].values, close_prices, timeperiod=14)
        
        # 计算日收益率
        df['daily_return'] = df['收盘'].pct_change()
        df['daily_return'] = df['daily_return'].clip(lower=-0.99, upper=0.99)
        
        # 计算波动率
        df['volatility_20'] = df['daily_return'].rolling(window=20, min_periods=1).std() * np.sqrt(252)
        
        return df

    def preprocess_data(self, df):
        """数据预处理"""
        # 处理缺失值
        df = df.ffill()  # 向前填充
        df = df.bfill()  # 向后填充剩余的NA
        df = df.fillna(0)  # 填充任何剩余的NA为0
        
        # 确保数值类型正确
        numeric_columns = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '换手率', '涨跌幅']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

    def analyze_stock(self, stock_code, start_date=None, end_date=None):
        """分析股票数据的主函数"""
        try:
            # 检查并更新数据
            self.update_if_needed()
            
            # 获取数据
            df = self.get_stock_data(stock_code, start_date, end_date)
            
            # 预处理数据
            df = self.preprocess_data(df)
            
            # 计算技术指标
            df = self.calculate_indicators(df)
            
            # 更新进度
            self.__class__.update_progress()
            
            return df
        except Exception as e:
            logging.debug(f"处理股票{stock_code}时出错: {str(e)}")
            # 即使出错也要更新进度
            self.__class__.update_progress()
            return None

def get_stock_analysis(stock_code, start_date=None, end_date=None):
    """便捷函数,用于快速获取股票分析结果"""
    analyzer = StockAnalyzer()
    return analyzer.analyze_stock(stock_code, start_date, end_date)
