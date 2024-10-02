from hatespeech.logger import *
from hatespeech.exception import *

logging.info("Welcome to out Project")

# Example of how you might use this CustomException class
try:
    1 / 0  # Example of an error (division by zero)
except Exception as e:
    raise CustomException("An error occurred", sys)