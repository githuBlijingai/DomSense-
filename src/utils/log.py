import logging
import sys
from pathlib import Path


class Logger:
    _instance = None

    def __new__(cls, name="jev_mdp", log_dir=None, level=logging.INFO):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, name="jev_mdp", log_dir=None, level=logging.INFO):
        if self._initialized:
            return
        self._initialized = True
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()

        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if log_dir:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(str(log_path / "log.txt"), encoding="utf-8")
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def get_logger(self):
        return self.logger

    @classmethod
    def get(cls):
        if cls._instance is None:
            return cls().get_logger()
        return cls._instance.get_logger()
