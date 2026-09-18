"""Private API lifecycle for single-service Streamlit deployments."""

import atexit
import socket
import threading
import time

import uvicorn

from app.main import create_app
from src.database import Database


class StorageConnectionError(RuntimeError):
    """A safe, user-facing storage failure code without credentials or host details."""


def connection_error_code(error):
    message = str(error).lower()
    if "password authentication failed" in message or "authentication failed" in message:
        return "authentication"
    if "could not translate host" in message or "name or service not known" in message:
        return "dns"
    if "network is unreachable" in message or "no route to host" in message:
        return "network_route"
    if "timeout" in message or "timed out" in message:
        return "timeout"
    if "ssl" in message or "certificate" in message:
        return "tls"
    return "connection"


class LocalAPI:
    def __init__(self, database_url):
        self.database = Database(database_url)
        try:
            self.database.initialize()
            self.database.healthy()
        except Exception as error:
            self.database.engine.dispose()
            raise StorageConnectionError(connection_error_code(error)) from None
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(("127.0.0.1", 0))
        self.url = f"http://127.0.0.1:{self.socket.getsockname()[1]}"
        self.server = uvicorn.Server(
            uvicorn.Config(
                create_app(self.database),
                host="127.0.0.1",
                log_level="critical",
                access_log=False,
                timeout_graceful_shutdown=3,
            )
        )
        self.thread = threading.Thread(
            target=self.server.run,
            kwargs={"sockets": [self.socket]},
            daemon=True,
            name="forecast-api",
        )
        self.thread.start()
        deadline = time.monotonic() + 30
        while not self.server.started:
            if not self.thread.is_alive() or time.monotonic() >= deadline:
                self.close()
                raise RuntimeError("Forecast service could not connect to storage")
            time.sleep(0.05)
        atexit.register(self.close)

    def close(self):
        self.server.should_exit = True
        self.thread.join(timeout=4)
        if not self.thread.is_alive():
            self.socket.close()
            self.database.engine.dispose()


_services = {}
_startup_lock = threading.Lock()


def local_api(database_url):
    """One loopback-only API per dashboard process, shared across reruns/sessions."""
    with _startup_lock:
        service = _services.get(database_url)
        if service is None or not service.thread.is_alive():
            service = LocalAPI(database_url)
            _services[database_url] = service
        return service
