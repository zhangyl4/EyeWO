import sys
import time
import traceback
import importlib
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('caption2qa_error.log'),
        logging.StreamHandler()
    ]
)

def run_with_retry(max_retries=500, delay=60):
    """
    运行主程序，发生错误时进行重试
    
    Args:
        max_retries (int): 最大重试次数
        delay (int): 重试之间的延迟时间(秒)
    """
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # 重新导入模块以确保每次都使用最新的代码
            module = importlib.import_module('judge_relative_v2')
            # 清除之前导入的模块
            if 'judge_relative_v2' in sys.modules:
                del sys.modules['judge_relative_v2']
            logging.info(f"开始运行 (尝试 {retry_count + 1}/{max_retries})")
            return
            
        except Exception as e:
            retry_count += 1
            error_msg = traceback.format_exc()
            logging.error(f"发生错误:\n{error_msg}")
            
            if retry_count < max_retries:
                logging.info(f"将在 {delay} 秒后重试...")
                time.sleep(delay)
            else:
                logging.error(f"达到最大重试次数 ({max_retries})，程序终止")
                raise

if __name__ == "__main__":
    run_with_retry() 