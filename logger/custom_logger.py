import os
import logging
from datetime import datetime

import structlog


class CustomLogger:

    def __init__(self, log_dir="logs"):

        # Create logs directory
        self.logs_dir = os.path.join(os.getcwd(), log_dir)
        os.makedirs(self.logs_dir, exist_ok=True)

        # Create timestamped log file
        log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
        self.log_file_path = os.path.join(self.logs_dir, log_file)

        # Configure logger only once
        self._configure_logging()

    def _configure_logging(self):

        # -----------------------------
        # File Handler
        # -----------------------------
        file_handler = logging.FileHandler(
            self.log_file_path,
            encoding="utf-8"
        )

        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(message)s")
        )

        # -----------------------------
     # -----------------------------
        console_handler = logging.StreamHandler()

        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter("%(message)s")
        )      # Console Handler
     

        # -----------------------------
        # Standard Python Logging
        # -----------------------------
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                console_handler,
                file_handler
            ],
            force=True
        )

        # -----------------------------
        # Structlog Configuration
        # -----------------------------
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(
                    fmt="iso",
                    utc=True,
                    key="timestamp"
                ),

                structlog.processors.add_log_level,

                structlog.processors.EventRenamer(
                    to="event"
                ),

                structlog.processors.JSONRenderer()
            ],

            logger_factory=structlog.stdlib.LoggerFactory(),

            cache_logger_on_first_use=True
        )

    def get_logger(self, name=__file__):

        logger_name = os.path.basename(name)

        return structlog.get_logger(logger_name)