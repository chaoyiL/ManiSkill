import threading
from typing import Any

import uvicorn
from fastapi import FastAPI


class RobotClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        self.host = host
        self.port = port

        self.app = FastAPI()
        self._lock = threading.Lock()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

        self.config: Any = None
        self.state: Any = None
        self.obs: Any = None
        self.action: Any = None

        self._register_routes()

    def _register_routes(self) -> None:
        @self.app.post("/message")
        def receive_message(data: dict):
            return self._handle_message(data)

    def _handle_message(self, data: dict):
        head = data["head"]
        content = data["content"]
        print(f"[robot client] HEAD {head} 收到消息")

        with self._lock:
            if head == "config":
                self.config = content
                response = {
                    "content":"get"
                }
                return response
            if head == "state":
                self.state = content
                return None
            if head == "obs":
                return self.obs
            if head == "action":
                self.action = content
                return None

        return {"error": f"unknown head: {head}"}

    def run(self) -> None:
        config = uvicorn.Config(self.app, host=self.host, port=self.port)
        self._server = uvicorn.Server(config)
        self._server.run()

    def start_background(self, daemon: bool = True) -> threading.Thread:
        if self._thread is not None and self._thread.is_alive():
            return self._thread

        self._thread = threading.Thread(
            target=self.run,
            name=f"RobotClient-{self.port}",
            daemon=daemon,
        )
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def get_value(self, name: str) -> Any:
        with self._lock:
            ans = getattr(self, name)
            setattr(self, name, None) # 获得值后立刻复位成None
            return ans

    def set_obs(self, obs: Any) -> None:
        with self._lock:
            self.obs = obs
