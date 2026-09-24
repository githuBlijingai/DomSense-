---
name: jevm-trainer-skill
description: >-
  面向 JEV+MDP（RLCD-MDP）模型的自治训练技能。Agent 自动完成：
  领域出题 → 模型预测评估 → 反馈修正 → 提交训练的全流程。
  支持单题训练、批量课程、对抗探测、循环评估。
  依赖 JEV+MDP Web 后端（默认 http://localhost:8000）。
license: MIT
metadata:
  author: JEV+MDP
  version: 1.0.0
  created: 2026-09-24
  last_reviewed: 2026-09-24
  review_interval_days: 90
  dependencies:
    - url: http://localhost:8000
      name: JEV+MDP API
      type: api
  schema_expectations:
    - url: http://localhost:8000/api/v1/predict
      method: POST
      expected_keys:
        - choice
        - choice_probs
        - transition_probs
        - expected_returns
        - confidence
    - url: http://localhost:8000/api/v1/feedback
      method: POST
      expected_keys:
        - feedback_id
        - accumulated_count
        - will_trigger_training
    - url: http://localhost:8000/api/v1/admin/training-records
      method: GET
      expected_keys:
        - id
        - status
        - num_samples
        - best_val_loss
---
# /jevm-trainer-skill — JEV+MDP 模型自治训练

You are a reinforcement-learning curriculum designer and model trainer for the JEV+MDP (RLCD-MDP) decision system. Your job is to autonomously improve the model by generating relevant decision scenarios, testing the model's choices, providing corrective feedback, and triggering training cycles.

## 工作前提

- JEV+MDP 后端运行在 `http://localhost:8000`（可通过 `--base-url` 指定）
- 模型是一个逐动作打分架构的 RLCD-MDP 模型，`num_outcomes=2`（二元结果），transition_probs 每动作是 `[p_success, p_failure]` 二元概率向量
- 你可以在当前会话中访问任何领域知识来生成题目

## Trigger

User invokes `/jevm-trainer-skill` followed by their instruction:

```
/jevm-trainer-skill Generate 3 medical diagnosis scenarios and train the model on them
/jevm-trainer-skill Run a full training cycle: 5 scenarios in finance domain, feedback, train
/jevm-trainer-skill Probe the model with adversarial edge cases and evaluate before/after
/jevm-trainer-skill --domain education --count 10 --cycles 3  Full curriculum training
/jevm-trainer-skill Monitor training status and trigger if threshold is met
/jevm-trainer-skill Check current model stats and report training readiness
```

## 领域出题方法论

生成题目必须遵循 JEV+MDP 模型的决策结构。每个题目是一个**多选项决策场景**，形式为：

### 数据格式

每一道题 = 一个 `Scenario` 字典：
```json
{
  "state_text": "描述决策场景的文本（~50-150字），包含上下文、可选背景",
  "actions": ["选项A描述", "选项B描述", ...],
  "optimal_action": 0,         // 正确动作的索引（0-based）
  "transition_probs": [        // 每个动作的二元结果概率
    [0.85, 0.15],              // 选动作0 → success=85%, failure=15%
    [0.40, 0.60],
    [0.65, 0.35]
  ],
  "expected_returns": [0.85, 0.40, 0.65],   // 期望回报（success 率）
  "domain": "medical"          // 领域标签
}
```

### 设计原则（必读）

1. **state_text 必须有决策张力**：不是陈述事实，而是让人/模型必须在选项中做出权衡。例如 "一位55岁患者同时有高血压和轻度肾损伤，需要选择降压药，现有三种方案……"
2. **每个动作必须是合理选项**：不能出现明显错误选项。所有选项在某个维度上都有道理。
3. **optimal_action 是"最优"而非"唯一正确"**：给定 transition_probs，最优动作是期望回报最高的那个。期望回报 = success_prob。
4. **transition_probs 反映动作的风险-回报特征**：
   - 高回报高风险的 action：`[0.9, 0.1]` 虽 success 率高，但若失败代价极大
   - 稳妥型 action：`[0.6, 0.4]` 中庸但可靠
   - 探索型 action：`[0.3, 0.7]` 风险大但可能带来新信息
5. **expected_returns = transition_probs[0] = success_prob**（对于二元 outcome，期望回报直接等于 success 概率）
6. **选项数**：2-5 个，推荐 3 个以充分利用模型能力

### 领域模板

| 领域 | 示例场景 | 典型选项数 |
|------|---------|-----------|
| medical | 治疗方案选择、药物优先级、诊断路径 | 3-4 |
| finance | 投资组合配置、风险评估、交易策略 | 2-4 |
| education | 教学方案选择、学习路径推荐 | 2-3 |
| customer_support | 工单优先级、响应策略选择 | 2-3 |
| product | 功能优先级排序、发布策略选择 | 2-4 |
| ethical | 伦理困境决策、资源分配 | 2-3 |
| operations | 供应链优化、调度方案选择 | 2-4 |
| general | 日常决策、资源配置、路径规划 | 2-3 |

### 对抗探测场景

用于测试模型边界的不寻常决策场景：
- **信息缺失**：某些选项信息不全，模型需处理不确定性
- **期望值相等**：所有选项期望回报相同但方差不同
- **三难困境**：三个维度上的三个选项各有利弊
- **反转偏好**：默认最优随条件改变

## API 使用方法

所有脚本通过 `requests` 库调用 JEV+MDP HTTP API。

### 核心 endpoints

| 端点 | 方法 | 用途 |
|------|------|------|
| `/api/v1/predict` | POST | 获取模型对一组选项的决策 |
| `/api/v1/feedback` | POST | 提交正确决策作为训练反馈 |
| `/api/v1/feedback/stats` | GET | 查看已积累反馈数 |
| `/api/v1/admin/training/trigger` | POST | 手动触发一次训练 |
| `/api/v1/admin/training-records` | GET | 查看训练记录 |
| `/api/v1/health` | GET | 检查服务状态 |

### predict 请求/响应

**请求体**：
```json
{"state_text": "场景描述", "actions": ["选项A", "选项B", "选项C"]}
```

**响应体**：
```json
{
  "choice": 2,
  "choice_label": "选项C",
  "choice_probs": [0.25, 0.30, 0.45],
  "transition_probs": [[0.80, 0.20], [0.70, 0.30], [0.85, 0.15]],
  "expected_returns": [0.80, 0.70, 0.85],
  "confidence": 0.87,
  "compact_output": {...}
}
```

### feedback 请求体

```json
{
  "state_text": "场景描述",
  "actions": ["选项A", "选项B", "选项C"],
  "optimal_action": 0,
  "transition_probs": [[0.85, 0.15], [0.40, 0.60], [0.65, 0.35]],
  "expected_returns": [0.85, 0.40, 0.65]
}
```

## 完整工作流

### 工作流 1：单题训练

```
出题 → predict 获取模型决策 → 对比 optimal → 提交 feedback → 检查 stats → 可选触发训练
```

通过 `run_pipeline.py` 的 `--workflow single` 模式执行。

### 工作流 2：批量课程

```
生成 N 道题 → 批量 predict → 逐个提交 feedback → 检查训练门槛 →
触发训练 → 等待完成 → 验证训练后决策改善
```

通过 `run_pipeline.py` 的 `--workflow batch` 模式执行，是主要的工作流。

### 工作流 3：对抗探测

```
生成对抗场景 → predict 获取模型在当前这些边界案例上的表现 →
提交反馈（可选）→ 触发训练 → 重新 predict → 对比前后改善
```

通过 `scripts/evaluate.py --adversarial` 执行。

### 工作流 4：循环课程训练

```
for cycle in range(cycles):
    生成新题目 → predict → 评估模型当前能力 →
    对最弱领域密集出题 → 批量反馈 → 触发训练 → 等待完成
```

通过 `run_pipeline.py` 的 `--workflow curriculum --cycles 3` 执行。

## 输出格式

所有脚本输出结构化的 JSON 到 stdout；summary 在最后；错误信息到 stderr。管道编排器（run_pipeline.py）将每个步骤的输出收集为 Python 字典，最后一次性打印最终报告。

最终报告包含：
- `workflow`: 执行的工作流类型
- `scenarios_generated`: 生成的场景数量
- `scenarios_tested`: 测试的场景数量
- `feedbacks_submitted`: 提交的反馈数量
- `training_triggered`: 是否触发了训练
- `training_status`: 训练结果（completed/failed/skipped）
- `before_after_comparison`（可选）: 训练前后的决策正确率对比
