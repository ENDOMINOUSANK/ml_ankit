import logging
import os
from datetime import datetime

# Define the log file name with a timestamp
log_file_name = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
# Define the directory where log files will be stored
logs_dir = os.path.join(os.getcwd(), "logs")
# Ensure the directory exists
os.makedirs(logs_dir, exist_ok=True)
# Full path to the log file
log_file_path = os.path.join(logs_dir, log_file_name)

logging.basicConfig(
    filename=log_file_path, 
    level=logging.INFO, 
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
)