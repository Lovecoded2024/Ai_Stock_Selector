import unittest
import asyncio
import pandas as pd
import os
from main import (
    get_all_stocks,
    get_stock_data_async,
    process_stock_batch,
    process_detailed_analysis_batch,
    generate_report
)
from news_sentiment import (
    get_stock_news_async,
    analyze_news_async,
    get_news_sentiment_async
)

class TestMainFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """测试开始前的设置"""
        # 确保report目录存在
        if not os.path.exists('report'):
            os.makedirs('report')
            
    def setUp(self):
        """每个测试用例开始前的设置"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
    def tearDown(self):
        """每个测试用例结束后的清理"""
        self.loop.close()
        
    def test_get_all_stocks(self):
        """测试获取所有股票代码"""
        try:
            stocks = get_all_stocks()
            self.assertIsInstance(stocks, list)
            self.assertTrue(len(stocks) > 0)
            # 验证股票代码格式
            for code in stocks:
                self.assertTrue(len(code) == 6)
                self.assertTrue(code.isdigit())
            print(f"成功获取{len(stocks)}只股票代码")
        except Exception as e:
            self.fail(f"获取股票代码失败: {str(e)}")
            
    def test_get_stock_data(self):
        """测试获取股票数据"""
        try:
            # 获取前5只股票的数据进行测试
            stocks = get_all_stocks()[:5]
            stock_data = self.loop.run_until_complete(get_stock_data_async(stocks))
            
            # 验证返回的数据
            self.assertIsInstance(stock_data, pd.DataFrame)
            self.assertTrue(len(stock_data) > 0)
            
            # 验证必要的列是否存在
            required_columns = ['代码', '日期', '收盘', 'MA_5', 'MA_10', 'MA_20', 'RSI_6', 'MACD']
            for col in required_columns:
                self.assertIn(col, stock_data.columns)
                
            print(f"成功获取{len(stocks)}只股票的数据")
            # 打印第一只股票的最新数据作为示例
            latest_data = stock_data[stock_data['代码'] == stocks[0]].iloc[0]
            print(f"股票{stocks[0]}最新数据示例:")
            print(latest_data)
        except Exception as e:
            self.fail(f"获取股票数据失败: {str(e)}")
            
    def test_stock_rating(self):
        """测试股票评分功能"""
        try:
            # 获取一只股票的数据进行测试
            stocks = get_all_stocks()[:1]
            stock_data = self.loop.run_until_complete(get_stock_data_async(stocks))
            
            # 进行评分
            ratings = self.loop.run_until_complete(process_stock_batch(stocks, stock_data))
            
            # 验证评分结果
            self.assertTrue(len(ratings) > 0)
            rating = ratings[0]
            self.assertIn('代码', rating)
            self.assertIn('评分', rating)
            self.assertIn('分析', rating)
            self.assertIn('建议', rating)
            
            # 验证评分范围
            self.assertTrue(0 <= rating['评分'] <= 100)
            # 验证建议是否为有效值
            self.assertIn(rating['建议'], ['买入', '持有', '卖出'])
            
            print(f"股票{rating['代码']}的评分结果:")
            print(f"评分: {rating['评分']}")
            print(f"建议: {rating['建议']}")
            print(f"分析: {rating['分析']}")
        except Exception as e:
            self.fail(f"股票评分测试失败: {str(e)}")
            
    def test_detailed_analysis(self):
        """测试深度分析功能"""
        try:
            # 获取一只股票的数据进行测试
            stocks = get_all_stocks()[:1]
            stock_data = self.loop.run_until_complete(get_stock_data_async(stocks))
            
            # 首先获取评分
            ratings = self.loop.run_until_complete(process_stock_batch(stocks, stock_data))
            
            # 进行深度分析
            analyses = self.loop.run_until_complete(
                process_detailed_analysis_batch(ratings, stock_data)
            )
            
            # 验证分析结果
            self.assertTrue(len(analyses) > 0)
            analysis = analyses[0]
            
            # 验证必要的字段是否存在
            required_fields = ['代码', '技术面分析', '基本面分析', '新闻舆情', '交易建议', '风险提示']
            for field in required_fields:
                self.assertIn(field, analysis)
                self.assertIsNotNone(analysis[field])
                
            print(f"股票{analysis['代码']}的深度分析结果:")
            for field in required_fields:
                if field != '代码':
                    print(f"\n{field}:")
                    print(analysis[field])
        except Exception as e:
            self.fail(f"深度分析测试失败: {str(e)}")
            
    def test_report_generation(self):
        """测试报告生成功能"""
        try:
            # 获取前3只股票的数据进行测试
            stocks = get_all_stocks()[:3]
            stock_data = self.loop.run_until_complete(get_stock_data_async(stocks))
            
            # 获取评分
            ratings = self.loop.run_until_complete(process_stock_batch(stocks, stock_data))
            
            # 进行深度分析
            analyses = self.loop.run_until_complete(
                process_detailed_analysis_batch(ratings, stock_data)
            )
            
            # 生成报告
            report_path = generate_report(ratings, analyses)
            
            # 验证报告文件是否生成
            self.assertTrue(os.path.exists(report_path))
            
            # 验证报告内容
            with open(report_path, 'r', encoding='utf-8') as f:
                report_content = f.read()
                
            # 验证报告是否包含必要的部分
            self.assertIn('股票筛选报告', report_content)
            self.assertIn('基础分析', report_content)
            self.assertIn('深度分析', report_content)
            
            print(f"报告已生成: {report_path}")
            print("报告预览:")
            print(report_content[:500] + "...")  # 只显示前500个字符
        except Exception as e:
            self.fail(f"报告生成测试失败: {str(e)}")
            
    def test_news_sentiment(self):
        """测试新闻舆情分析功能"""
        try:
            # 获取一只股票的新闻数据
            stocks = get_all_stocks()[:1]
            stock_code = stocks[0]
            
            # 1. 测试新闻获取
            news_df = self.loop.run_until_complete(get_stock_news_async(stock_code))
            if news_df is not None:
                self.assertIsInstance(news_df, pd.DataFrame)
                required_columns = ['发布时间', '新闻标题', '新闻内容']
                for col in required_columns:
                    self.assertIn(col, news_df.columns)
                print(f"\n获取到{len(news_df)}条新闻")
                if len(news_df) > 0:
                    print("最新一条新闻：")
                    latest_news = news_df.iloc[0]
                    print(f"标题：{latest_news['新闻标题']}")
                    print(f"时间：{latest_news['发布时间']}")
            
            # 2. 测试新闻分析
            if news_df is not None and not news_df.empty:
                analysis = self.loop.run_until_complete(analyze_news_async(news_df))
                self.assertIsInstance(analysis, str)
                self.assertNotEqual(analysis, "处理新闻内容时出错")
                print("\n新闻分析结果：")
                print(analysis)
            
            # 3. 测试完整的新闻舆情分析流程
            sentiment = self.loop.run_until_complete(get_news_sentiment_async(stock_code))
            self.assertIsInstance(sentiment, str)
            self.assertNotEqual(sentiment, "无法获取新闻舆情分析")
            print("\n完整舆情分析结果：")
            print(sentiment)
            
        except Exception as e:
            self.fail(f"新闻舆情分析测试失败: {str(e)}")
            
    def test_full_process(self):
        """测试完整的处理流程"""
        try:
            # 获取前5只股票进行完整流程测试
            stocks = get_all_stocks()[:5]
            
            # 1. 获取数据
            stock_data = self.loop.run_until_complete(get_stock_data_async(stocks))
            self.assertTrue(len(stock_data) > 0)
            
            # 2. 评分
            ratings = self.loop.run_until_complete(process_stock_batch(stocks, stock_data))
            self.assertEqual(len(ratings), len(stocks))
            
            # 3. 深度分析
            analyses = self.loop.run_until_complete(
                process_detailed_analysis_batch(ratings, stock_data)
            )
            self.assertTrue(len(analyses) > 0)
            
            # 4. 生成报告
            report_path = generate_report(ratings, analyses)
            self.assertTrue(os.path.exists(report_path))
            
            print("\n完整流程测试结果:")
            print(f"处理的股票数量: {len(stocks)}")
            print(f"生成的报告路径: {report_path}")
            
            # 打印评分最高的股票
            top_stock = max(ratings, key=lambda x: x['评分'])
            print(f"\n评分最高的股票:")
            print(f"代码: {top_stock['代码']}")
            print(f"评分: {top_stock['评分']}")
            print(f"建议: {top_stock['建议']}")
        except Exception as e:
            self.fail(f"完整流程测试失败: {str(e)}")

def run_tests():
    """运行所有测试"""
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMainFunctions)
    # 运行测试
    unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == '__main__':
    run_tests()
