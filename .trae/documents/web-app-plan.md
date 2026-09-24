# Web 应用层实现计划 — JEV+MDP System One 决策模型在线服务

## Context

用户希望让客户通过网页使用已训练好的决策模型，并具备：
1. **反馈机制** — 用户点"结果有误"即可提交标注数据
2. **自动训练** — 反馈积累到 50 条后后台自动微调，无需手动执行训练命令
3. **监督学习双入口** — 管理员页面上传 CSV + REST API 接口
4. **标注文件模板下载 + 说明网页** — 让客户知道标注格式

技术选型：FastAPI 后端 + 独立前端，SQLite 存储反馈，积累触发训练策略。

## 文件结构（新建文件）

```
web/
├── backend/
│   ├── main.py                # FastAPI 入口、路由注册、startup 钩子
│   ├── config.py              # Web 层配置（端口、阈值 50、checkpoint 路径）
│   ├── database.py            # SQLite schema 初始化（feedbacks + training_records 表）
│   ├── schemas.py             # Pydantic 请求/响应模型
│   ├── model_manager.py       # 模型单例 + RLock 热更新
│   ├── feedback_store.py      # 反馈行 → MDPDataPoint → DataLoader 转换
│   ├── training_worker.py     # 后台训练线程 + task_queue
│   └── routes/
│       ├── predict.py         # POST /api/v1/predict
│       ├── feedback.py        # POST /api/v1/feedback + GET /stats
│       ├── admin.py           # POST /upload-csv + POST /train + GET /training-records
│       └── templates.py       # GET /api/v1/template/annotation.csv 下载
├── frontend/static/
│   ├── index.html             # 用户决策页（输入→结果→反馈）
│   ├── admin.html             # 管理员页（上传+训练+记录+模型信息）
│   ├── annotation-guide.html  # 标注说明页（字段表+模板+流程图）
│   ├── css/main.css           # 科技感配色 #0a0e27 + #00d4ff
│   ├── css/particles.css      # 全屏粒子背景
│   ├── js/api.js              # fetch 封装
│   ├── js/app.js              # 用户页逻辑
│   ├── js/admin.js            # 管理员页逻辑
│   ├── js/particles.min.js    # 离线粒子库
│   └── assets/annotation_template.csv
scripts/
└── serve.ps1                  # uvicorn 启动脚本
data/                          # 运行时自动创建
├── feedback.db
└── uploaded_csv/
```

## API 路由

| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/api/v1/predict` | 推理：输入文本+动作 → 返回决策/概率/回报/置信度 |
| POST | `/api/v1/feedback` | 反馈：提交正确答案 → 入库，达 50 条触发训练 |
| GET | `/api/v1/feedback/stats` | 反馈统计：总数/未用/距下次训练条数 |
| POST | `/api/v1/admin/upload-csv` | 上传 CSV 标注数据 → 解析校验入库 |
| POST | `/api/v1/admin/train` | 手动触发训练（可选指定 feedback_ids 和 max_steps） |
| GET | `/api/v1/admin/training-records` | 训练记录列表 |
| GET | `/api/v1/admin/model/info` | 当前模型版本/checkpoint/loss |
| GET | `/api/v1/template/annotation.csv` | 下载 CSV 模板 |
| GET | `/` | 用户决策页 |
| GET | `/admin` | 管理员页 |
| GET | `/annotation-guide` | 标注说明页 |

## SQLite Schema

**feedbacks 表**：id, state_text, actions(JSON), optimal_action, transition_probs(JSON), expected_returns(JSON), predicted_choice, source(web/csv_upload/api), created_at, used_in_training(0/1)

**training_records 表**：id, run_name, trigger_source(auto_threshold/manual/csv_upload), num_samples, status(pending/running/completed/failed), start_time, end_time, best_val_loss, checkpoint_path, feedback_ids(JSON), log, error_message

## 关键机制

### 模型热更新 (`model_manager.py`)
- 模型单例 + `threading.RLock` 保护
- `predict()` 锁内取引用 → 锁外 `torch.no_grad()` 执行 forward
- `swap_model()` 训练完成后替换引用 + version+1
- 训练在后台线程用全新 SystemOneModel 实例，不接触在线模型

### 自动训练 (`training_worker.py`)
- 后台 daemon 线程 + `queue.Queue` 任务队列
- 每次反馈入库后检查 `COUNT(*) WHERE used_in_training=0`
- 达 50 条 → enqueue 训练任务
- 训练流程：反馈 → MDPDataPoint → MDPDataset → DataLoader → RLCDMDPTrainer(200 步) → 保存 checkpoint → swap_model → 标记 used_in_training=1
- 手动/自动互斥：执行前检查有无 status='running'

### 反馈→训练数据转换 (`feedback_store.py`)
- `feedback_row_to_datapoint()` 将 DB 行转为 `MDPDataPoint`
- `build_dataloader()` 构建 DataLoader，复用 `MDPDataset.collate_fn`
- 校验：actions 长度一致、transition_probs 行和=1.0、optimal_action 在范围内

## 标注 CSV 模板

```csv
state_text,actions,optimal_action,transition_probs,expected_returns
"决策场景描述","选项A|选项B|选项C",0,"0.7,0.3|0.4,0.6|0.2,0.8","5.5,3.2,7.8"
```

字段：state_text(场景), actions(竖线分隔), optimal_action(0-based 索引), transition_probs(动作间竖线/动作内逗号), expected_returns(逗号分隔)

## 前端设计

- **科技感**：#0a0e27 深夜蓝背景 + #00d4ff 青色 + #7c3aed 紫渐变
- **全屏粒子背景**：particles.js 离线本地副本，粒子连线+鼠标吸附
- **按钮风格**：2px 边框 + 发光 shadow，放在内容卡片外部
- **用户页**：输入区 → 结果区（概率条形图+置信度）→ 反馈区（选择正确答案+提交）
- **管理员页**：CSV 拖拽上传 + 手动训练按钮 + 训练记录表 + 模型信息卡 + 反馈进度环
- **标注说明页**：字段说明表 + CSV 模板代码块 + 流程示意图 + 下载按钮

## 依赖追加

```
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
python-multipart>=0.0.6
aiofiles>=23.2.0
```

## 实现顺序

1. `database.py` + `schemas.py` — 无外部依赖
2. `model_manager.py` + `feedback_store.py` — 对接已有 SystemOneModel / MDPDataset
3. `routes/predict.py` + `routes/feedback.py` — 验证推理与反馈链路
4. `training_worker.py` — 后台训练线程
5. `routes/admin.py` + `routes/templates.py` — 管理员功能
6. 前端三页 + CSS + JS（粒子背景、用户页、管理员页、标注说明页）
7. `scripts/serve.ps1` 启动脚本
8. 更新 `requirements.txt` + `README.md`

## 验证

```bash
# 启动
.\scripts\serve.ps1
# 用户页
http://localhost:8000/
# 管理员页
http://localhost:8000/admin
# 标注说明页
http://localhost:8000/annotation-guide
# 推理测试
curl -X POST http://localhost:8000/api/v1/predict -H "Content-Type: application/json" -d '{"state_text":"选择方案","actions":["A","B","C"]}'
# 反馈测试
curl -X POST http://localhost:8000/api/v1/feedback -H "Content-Type: application/json" -d '{"state_text":"选择方案","actions":["A","B","C"],"optimal_action":1,"transition_probs":[[0.5,0.5],[0.3,0.7],[0.6,0.4]],"expected_returns":[3,5,4]}'
# 查看统计
curl http://localhost:8000/api/v1/feedback/stats
```

## 关键复用

- `src/models/system_one_model.py` — SystemOneModel.forward() + generate_compact_output()
- `src/data/synthetic.py` — MDPDataPoint + MDPDataset + collate_fn
- `src/training/rlcd_mdp_trainer.py` — RLCDMDPTrainer.train_step()
- `config/base.yaml` — 模型/训练配置
- `src/utils/log.py` — Logger
