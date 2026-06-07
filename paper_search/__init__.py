from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(".env.local"), override=True)
load_dotenv(dotenv_path=Path(".env"), override=False)

__all__ = ["__version__"]

__version__ = "0.1.0"
