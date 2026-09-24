"""FastAPI 应用入口。

启动：
    python -m web.backend.main
或：
    uvicorn web.backend.main:app --host 0.0.0.0 --port 8000

路由前缀：/api/v1
"""
import os

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .config import (
    CHECKPOINT_PATH,
    DB_PATH,
    HOST,
    OUTPUT_DIR,
    PORT,
    UPLOAD_DIR,
    load_merged_config,
)
from .database import init_database
from .model_manager import ModelManager
from .routes import admin, feedback, predict, templates
from .training_worker import TrainingWorker

app = FastAPI(title="域策 DomSense")

# 注册路由器（统一前缀 /api/v1）
_API_PREFIX = "/api/v1"
app.include_router(predict.router, prefix=_API_PREFIX)
app.include_router(feedback.router, prefix=_API_PREFIX)
app.include_router(admin.router, prefix=_API_PREFIX)
app.include_router(templates.router, prefix=_API_PREFIX)

# 挂载静态文件（前端资源位于 web/frontend/static）
_STATIC_DIR = "web/frontend/static"
if os.path.isdir(_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


def _serve_html(filename: str, fallback: str) -> FileResponse:
    """优先返回前端静态 HTML 文件，缺失时回退到内联 HTML。"""
    file_path = os.path.join(_STATIC_DIR, filename)
    if os.path.isfile(file_path):
        return FileResponse(file_path, media_type="text/html")
    return HTMLResponse(fallback)


@app.on_event("startup")
def on_startup() -> None:
    """启动初始化：创建目录、建表、加载模型、启动 worker。"""
    # 1. 确保目录存在
    for path in (
        os.path.dirname(DB_PATH),
        UPLOAD_DIR,
        OUTPUT_DIR,
        os.path.dirname(CHECKPOINT_PATH),
    ):
        if path:
            os.makedirs(path, exist_ok=True)

    # 2. 初始化数据库
    init_database()

    # 3. 加载模型
    merged = load_merged_config()
    model_config = dict(merged.get("model", {}))
    manager = ModelManager()
    manager.load_initial(CHECKPOINT_PATH, model_config)

    # 4. 启动训练 worker
    worker = TrainingWorker()
    worker.start()


@app.on_event("shutdown")
def on_shutdown() -> None:
    """关闭清理：停止 worker。"""
    worker = TrainingWorker()
    worker.stop()


@app.get("/", response_class=HTMLResponse)
def index():
    """首页：返回前端 index.html，缺失时回退到内联 HTML。"""
    return _serve_html(
        "index.html",
        "<html><head><meta charset='utf-8'><title>域策 DomSense</title></head>"
        "<body><h1>域策 DomSense</h1>"
        "<p>快速入口：</p><ul>"
        "<li><a href='/admin'>管理后台</a></li>"
        "<li><a href='/annotation-guide'>标注指南</a></li>"
        "<li><a href='/api/v1/model/info'>模型信息</a></li>"
        "<li><a href='/api/v1/stats'>反馈统计</a></li>"
        "</ul></body></html>",
    )


@app.get("/admin", response_class=HTMLResponse)
def admin_page():
    """管理后台页面：返回前端 admin.html，缺失时回退到内联 HTML。"""
    return _serve_html(
        "admin.html",
        "<html><head><meta charset='utf-8'><title>管理后台</title></head>"
        "<body><h1>管理后台</h1>"
        "<p>训练记录、模型信息、CSV 上传。</p><ul>"
        "<li><a href='/api/v1/model/info'>模型信息</a></li>"
        "<li><a href='/api/v1/training-records'>训练记录</a></li>"
        "<li><a href='/api/v1/stats'>反馈统计</a></li>"
        "<li><a href='/api/v1/template/annotation.csv'>下载 CSV 模板</a></li>"
        "</ul>"
        "<p>手动训练（POST /api/v1/train）与上传 CSV（POST /api/v1/upload-csv）"
        "需通过 HTTP 客户端调用。</p>"
        "</body></html>",
    )


@app.get("/annotation-guide", response_class=HTMLResponse)
def annotation_guide_page():
    """标注指南页面：返回前端 annotation-guide.html，缺失时回退到内联 HTML。"""
    return _serve_html(
        "annotation-guide.html",
        "<html><head><meta charset='utf-8'><title>标注指南</title></head>"
        "<body><h1>标注指南</h1>"
        "<p>反馈数据字段说明：</p><ul>"
        "<li><b>state_text</b>：决策场景文本（必填）</li>"
        "<li><b>actions</b>：可选动作列表，JSON 字符串（必填）</li>"
        "<li><b>optimal_action</b>：最优动作索引，0-based（必填）</li>"
        "<li><b>transition_probs</b>：转移概率矩阵，JSON 字符串，"
        "行和=1.0（必填）</li>"
        "<li><b>expected_returns</b>：期望回报列表，JSON 字符串（必填）</li>"
        "<li><b>predicted_choice</b>：模型预测选择索引（可空）</li>"
        "<li><b>source</b>：数据来源，默认 web（可空）</li>"
        "</ul>"
        "<p><a href='/api/v1/template/annotation.csv'>下载 CSV 模板</a> | "
        "<a href='/api/v1/template/annotation-guide'>查看 JSON 字段说明</a></p>"
        "</body></html>",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
