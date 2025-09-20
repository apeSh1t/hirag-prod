"""支持 `python -m colbert_remote_service` 直接启动服务。"""

import uvicorn

from .main import app


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()
