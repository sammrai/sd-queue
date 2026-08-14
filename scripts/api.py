from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import gradio as gr

from modules.api import api, models
from modules import script_callbacks
import modules.shared as shared
from secrets import compare_digest

from scripts.task_manager import TaskManager

import requests

version = "0.0.1"

task_manager = TaskManager()

def async_api(_: gr.Blocks, app: FastAPI):    
    if shared.cmd_opts.api_auth:
        opts_credentials = {}
        for auth in shared.cmd_opts.api_auth.split(","):
            user, password = auth.split(":")
            opts_credentials[user] = password
    
    def auth(credentials: HTTPBasicCredentials = Depends(HTTPBasic())):
        if credentials.username in opts_credentials:
            if compare_digest(credentials.password, opts_credentials[credentials.username]):
                return True

        raise HTTPException(status_code=401, detail="Incorrect username or password", headers={"WWW-Authenticate": "Basic"})

    def auth_required():
        return shared.cmd_opts.api_auth is not False and shared.cmd_opts.api_auth is not None
    
    def get_auth_dependency():
        if auth_required():
            return [Depends(auth)]
        return []
    
    @app.get("/sdapi/queue/login", dependencies=get_auth_dependency())
    async def login():
        return {"status": True, "version": version}
    
    @app.post("/sdapi/queue/txt2img", dependencies=get_auth_dependency())
    async def txt2imgapi(request: Request, txt2imgreq: models.StableDiffusionTxt2ImgProcessingAPI):
        route = next((route for route in request.app.routes if route.path == "/sdapi/v1/txt2img"), None)
        if route:
            task_id, success = task_manager.add_task(route.endpoint, txt2imgreq)
            if success:
                return {"status": "queued", "task_id": task_id}
            else:
                raise HTTPException(status_code=503, detail="Queue is full")
        return {"status": "error"}

    @app.get("/sdapi/queue/{task_id}/status", dependencies=get_auth_dependency())
    async def get_task_status(task_id: str, request: Request):
        task = task_manager.get_status(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        response = {"status": task["status"]}

        if "queue_position" in task:
            response["queue_position"] = task["queue_position"]

        if task["status"] == "completed":
            response["result"] = task["result"]
        elif task["status"] == "failed":
            # ワーカーが例外の str(e) を result に格納しているが、これまで返していなかった
            # ため、呼び出し側は failed の理由を知る手段が無かった。tasks_db はオンメモリで
            # 失敗時点を過ぎると消えるので、ここで返さないと原因は永久に失われる。
            response["error"] = task.get("result")
        elif task["status"] == "in-progress":
            route = next((route for route in request.app.routes if route.path == "/sdapi/v1/progress"), None)
            if route:
                progressreq = models.ProgressRequest(skip_current_image=False)
                info = route.endpoint(progressreq)
                response["progress"] = info.progress
                response["eta_relative"] = info.eta_relative
            else:
                print("Route /sdapi/v1/progress not found")
        
        return response

    @app.delete("/sdapi/queue/{task_id}/remove", dependencies=get_auth_dependency())
    async def remove_specific_task(task_id: str):
        result = task_manager.remove_specific_task(
            task_id, interrupt_callback=lambda: shared.state.interrupt()
        )
        if result == "removed":
            return {"status": "success", "message": f"Task {task_id} has been removed"}
        elif result == "interrupted":
            return {"status": "success", "message": f"Task {task_id} was in progress and has been interrupted"}
        elif result == "not_found":
            raise HTTPException(status_code=404, detail="Task not found")
        else:  # "not_removable"
            raise HTTPException(status_code=400, detail="Task cannot be removed in its current state")

script_callbacks.on_app_started(async_api)
