"""Web 应用端到端测试脚本"""
import sys
sys.path.insert(0, '.')

from web.backend.main import app
from fastapi.testclient import TestClient

with TestClient(app) as client:
    # 测试推理
    print("=== 推理测试 ===")
    resp = client.post("/api/v1/predict", json={
        "state_text": "选择技术方案",
        "actions": ["保守", "激进", "均衡"]
    })
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        d = resp.json()
        print(f"Choice: {d.get('choice')} ({d.get('choice_label')})")
        print(f"Confidence: {d.get('confidence')}")
        print(f"Probs: {d.get('choice_probs')}")
        print(f"Returns: {d.get('expected_returns')}")
        print(f"Model version: {d.get('model_version')}")
    else:
        print(f"Error: {resp.text[:500]}")

    # 测试反馈
    print("\n=== 反馈测试 ===")
    resp2 = client.post("/api/v1/feedback", json={
        "state_text": "选择技术方案",
        "actions": ["保守", "激进", "均衡"],
        "optimal_action": 1,
        "transition_probs": [[0.5, 0.5], [0.3, 0.7], [0.6, 0.4]],
        "expected_returns": [3, 5, 4],
        "predicted_choice": 0
    })
    print(f"Status: {resp2.status_code}")
    if resp2.status_code == 200:
        d2 = resp2.json()
        print(f"Feedback ID: {d2.get('feedback_id')}")
        print(f"Accumulated: {d2.get('accumulated_count')}")
        print(f"Will train: {d2.get('will_trigger_training')}")
        print(f"Next in: {d2.get('next_training_in')}")
    else:
        print(f"Error: {resp2.text[:500]}")

    # 测试统计
    print("\n=== 统计测试 ===")
    resp3 = client.get("/api/v1/feedback/stats")
    print(f"Status: {resp3.status_code}")
    if resp3.status_code == 200:
        print(resp3.json())
    else:
        print(f"Error: {resp3.text[:500]}")

    # 测试模型信息
    print("\n=== 模型信息 ===")
    resp4 = client.get("/api/v1/admin/model/info")
    print(f"Status: {resp4.status_code}")
    if resp4.status_code == 200:
        d4 = resp4.json()
        print(f"Version: {d4.get('version')}")
        print(f"Checkpoint: {d4.get('checkpoint_path')}")
        print(f"Params: {d4.get('num_params')}")
    else:
        print(f"Error: {resp4.text[:500]}")

    # 测试训练记录
    print("\n=== 训练记录 ===")
    resp5 = client.get("/api/v1/admin/training-records")
    print(f"Status: {resp5.status_code}")
    if resp5.status_code == 200:
        print(resp5.json())
    else:
        print(f"Error: {resp5.text[:500]}")

    # 测试 CSV 模板下载
    print("\n=== CSV 模板下载 ===")
    resp6 = client.get("/api/v1/template/annotation.csv")
    print(f"Status: {resp6.status_code}")
    print(f"Content-Type: {resp6.headers.get('content-type')}")

    # 测试 HTML 页面
    print("\n=== HTML 页面 ===")
    for path in ["/", "/admin", "/annotation-guide"]:
        r = client.get(path)
        print(f"GET {path}: {r.status_code} ({len(r.content)} bytes)")

    print("\n=== 全部 API 测试完成 ===")
