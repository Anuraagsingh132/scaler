"""
SecureAI-Guard: FastAPI environment server.
Exposes /reset, /step, /state and auxiliary endpoints.
"""
import logging
import os
import sys

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from env.engine import SecureAIGuardEngine
from tasks.registry import TaskRegistry
from graders.security_grader import SecurityGrader
from schema.models import Action, StepResponse, Observation, State

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("secureai-guard")

app = FastAPI(
    title="SecureAI-Guard API",
    description="Stateful POMDP for Autonomous Digital Defense — OpenEnv compliant",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
env = SecureAIGuardEngine()
task_registry = TaskRegistry()
grader = SecurityGrader()
_episode_rewards: List = []
_current_task_id: str = "basic_security"


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class ResetRequest(BaseModel):
    task_id: Optional[str] = "basic_security"
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: Action


class GradeRequest(BaseModel):
    task_id: Optional[str] = "basic_security"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "name": "SecureAI-Guard",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grade", "/preference_data", "/health"],
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/metadata")
async def metadata():
    """OpenEnv metadata endpoint for runtime validation."""
    return {
        "name": "SecureAI-Guard",
        "description": "Stateful POMDP for autonomous digital defense across SMS, email, and web.",
        "version": "1.0.0",
    }


@app.get("/schema")
async def schema():
    """Return action/observation/state schemas for OpenEnv runtime validation."""
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": State.model_json_schema(),
    }


@app.post("/mcp")
async def mcp(payload: Optional[Dict[str, Any]] = None):
    """Minimal JSON-RPC-compatible stub so runtime validators can reach /mcp."""
    request_id = payload.get("id") if isinstance(payload, dict) else None
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": -32601,
            "message": "MCP endpoint is not implemented for this environment.",
        },
    }


@app.post("/reset")
async def reset(request: ResetRequest):
    """Reset environment. Returns first observation."""
    global _episode_rewards, _current_task_id

    _current_task_id = request.task_id or "basic_security"
    _episode_rewards = []

    try:
        observation = env.reset(seed=request.seed, task_id=_current_task_id)
        logger.info("Episode reset | task=%s seed=%s", _current_task_id, request.seed)
        return {
            "observation": observation.model_dump(),
            "state": env.get_state().model_dump(),
            "task_id": _current_task_id,
        }
    except Exception as exc:
        logger.exception("Reset failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/step")
async def step(request: StepRequest):
    """Execute one environment step."""
    global _episode_rewards

    try:
        response: StepResponse = env.step(request.action)
        _episode_rewards.append(response.reward)

        result = response.model_dump()

        if response.done:
            grade = grader.grade_episode(response.state, _episode_rewards, _current_task_id)
            result["grade"] = grade.model_dump()
            logger.info(
                "Episode done | score=%.4f grade=%s steps=%d",
                grade.score, grade.grade, response.state.step_count,
            )

        return result
    except Exception as exc:
        logger.exception("Step failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/state")
async def get_state():
    """Get current environment state."""
    return env.get_state().model_dump()


@app.get("/tasks")
async def list_tasks():
    """List all available tasks."""
    return {"tasks": [t.model_dump() for t in task_registry.list_tasks()]}


@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """Get a specific task definition."""
    try:
        return task_registry.get_task(task_id).model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/grade")
async def grade_episode(request: GradeRequest):
    """Grade the current episode explicitly."""
    state = env.get_state()
    grade = grader.grade_episode(state, _episode_rewards, request.task_id or _current_task_id)
    return grade.model_dump()


@app.get("/preference_data")
async def get_preference_data():
    """Get logged DPO preference pairs."""
    return {"preference_pairs": env.get_preference_data()}


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")


@app.get("/dashboard")
async def dashboard():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="frontend")


if __name__ == "__main__":
    import webbrowser
    import threading

    def open_browser():
        """Open the dashboard in the default browser after a short delay."""
        import time
        time.sleep(1.5)
        webbrowser.open("http://localhost:7860/dashboard")

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("app:app", host="0.0.0.0", port=7860, log_level="info")
