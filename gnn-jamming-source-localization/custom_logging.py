import logging
from logging import Logger


class ColoredFormatter(logging.Formatter):
    COLORS: dict[str, str] = {
        'DEBUG': '\033[94m',   # Blue
        'INFO': '\033[97m',    # White
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',   # Red
        'CRITICAL': '\033[91m'  # Red
    }
    RESET: str = '\033[0m'

    def format(self, record) -> str:
        log_level: str = record.levelname
        log_color: str = self.COLORS.get(log_level, self.RESET)
        log_message: str = super().format(record)
        return f"{log_color}{log_message}{self.RESET}"


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    root_logger: Logger = logging.getLogger()
    colored_formatter: ColoredFormatter = ColoredFormatter(fmt='%(asctime)s - %(levelname)s - %(message)s')
    for handler in root_logger.handlers:
        handler.setFormatter(fmt=colored_formatter)
