import sqlite3
from abc import ABC, abstractmethod
from os import PathLike

from classconfig import ConfigurableValue, RelativePathTransformer

from sofairagent.api.base import APIRequest, APIOutput


class APILogger(ABC):
    """
    Base class for logging API requests and responses.
    """

    @abstractmethod
    def log_request(self, request: APIRequest):
        ...

    @abstractmethod
    def log_response(self, output: APIOutput):
        ...


class APISQLiteLogger(APILogger):
    """
    Logs API requests and responses to a SQLite database.
    """
    db_path: str = ConfigurableValue(
        "Path to the SQLite database file for logging.",
        transform=RelativePathTransformer()
    )

    def __init__(self, db_path: str | PathLike[str]):
        """
        Initializes the SQLiteSearchCache with the specified database path.
        """
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS api_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date_created TIMESTAMP DATETIME DEFAULT(datetime('subsec')),
                    type TEXT CHECK(type IN ('request', 'response')),
                    body TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def log_request(self, request: APIRequest):
        with self.conn:
            self.conn.execute(
                "INSERT INTO api_logs (type, body) VALUES (?, ?)",
                ("request", request.model_dump_json())
            )

    def log_response(self, output: APIOutput):
        with self.conn:
            self.conn.execute(
                "INSERT INTO api_logs (type, body) VALUES (?, ?)",
                ("response", output.model_dump_json())
            )

    def __del__(self):
        self.conn.close()
