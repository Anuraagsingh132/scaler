"""
OpenEnv validator entry point for SecureAI-Guard.

This thin wrapper exposes a conventional server/app.py with a main() function
while reusing the existing FastAPI application defined at the repository root.
"""

from pathlib import Path
import sys

import uvicorn


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import app  # noqa: E402


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")


if __name__ == "__main__":
    main()
