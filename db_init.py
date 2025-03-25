import sqlite3
import pandas as pd
import os
from datetime import datetime

def init_database():
    # 连接到SQLite数据库
    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()
    
    # 创建股票信息表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS stock_info (
        stock_code TEXT PRIMARY KEY,
        stock_name TEXT NOT NULL
    )
    ''')
    
    # 创建每日行情表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS daily_quote (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        stock_code TEXT NOT NULL,
        trade_date DATE NOT NULL,
        open_price REAL NOT NULL,
        close_price REAL NOT NULL,
        high_price REAL NOT NULL,
        low_price REAL NOT NULL,
        volume REAL NOT NULL,
        amount REAL NOT NULL,
        amplitude REAL NOT NULL,
        change_percent REAL NOT NULL,
        change_amount REAL NOT NULL,
        turnover_rate REAL NOT NULL,
        UNIQUE(stock_code, trade_date)
    )
    ''')
    
    conn.commit()
    return conn

def import_csv_data(conn, csv_path):
    # 从文件名解析股票代码和名称
    filename = os.path.basename(csv_path)
    stock_code = filename.split('_')[0]
    stock_name = filename.split('_')[1].replace('.csv', '')
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    cursor = conn.cursor()
    
    # 插入股票信息
    cursor.execute('INSERT OR REPLACE INTO stock_info (stock_code, stock_name) VALUES (?, ?)',
                  (stock_code, stock_name))
    
    # 插入每日行情数据
    for _, row in df.iterrows():
        cursor.execute('''
        INSERT OR REPLACE INTO daily_quote 
        (stock_code, trade_date, open_price, close_price, high_price, low_price, 
         volume, amount, amplitude, change_percent, change_amount, turnover_rate)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            stock_code,
            row['日期'],
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
    
    conn.commit()

def update_database():
    conn = init_database()
    
    # 遍历data目录下的所有CSV文件
    data_dir = 'data'
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            csv_path = os.path.join(data_dir, filename)
            import_csv_data(conn, csv_path)
    
    conn.close()

if __name__ == '__main__':
    update_database()
    print("数据库更新完成!")
