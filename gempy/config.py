import os

from dotenv import load_dotenv

# Define the paths for the .env files

script_dir = os.path.dirname(os.path.abspath(__file__))

dotenv_path = os.path.join(script_dir, '../.env')
dotenv_gempy_engine_path = os.path.expanduser('~/.env')

# Check if the .env files exist and prioritize the local .env file
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
elif os.path.exists(dotenv_gempy_engine_path):
    load_dotenv(dotenv_gempy_engine_path)
else:
    load_dotenv()
