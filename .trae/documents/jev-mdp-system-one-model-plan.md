# JEV+MDP：System One 决策链模型 论文复现计划（v2 - 深化版）

## 一、项目愿景（Summary）

从零构建一个**原创的 System One 决策链模型**，融合 Jev 式"闭集概率输出"理念与 **MDP（马尔可夫决策过程）** 完整多步框架，作为一个可发表论文的技术复现项目。

### 🎯 核心理论贡献（架构 + 训练统一贡献）

> **"隐式多步 MDP 前瞻 → 紧凑下一步输出"**

模型在**潜空间隐式展开多步决策链**（不产生显式 token），但对用户只暴露**紧凑的下一步最优决策 + 转移概率分布 + 期望回报**。这正是用户强调的**"用户输出只需后一步，但内部是完整多步 MDP"**的设计。

### 这一设计统一了三件事
1. **架构创新**：LLM 编码器 + 定制决策头 + **隐式前瞻 MDP Head**（新架构）
2. **采样创新**：并行采样器（一次前向输出所有决策答案，无自回归 token 生成）
3. **训练创新**：RLCD-MDP 统一框架——把**概率校准**定义为**状态转移概率估计**，并对**期望回报 Q(s,a)** 联合优化

---

## 二、当前状态分析

### 环境现状
- **Python**: 3.13.5 已安装
- **未安装**: transformers, torch, datasets（需全部安装）
- **平台**: Windows | **算力**: 无 GPU，CPU 训练
- **工作目录**: 完全空白的全新项目

### 环境约束
- Python 3.13 需 PyTorch ≥ 2.4+
- CPU 训练 → 小模型 + LoRA
- Windows 平台需注意训练库兼容性

### 已锁定技术决策
| 维度 | 选择 |
|------|------|
| 技术栈 | HF 后训练生态（transformers + TRL + PEFT） |
| 架构策略 | LLM 编码器 + 定制决策头 + 隐式前瞻 MDP Head |
| 采样式 | 并行采样器（多决策头并行） |
| MDP 融合 | 转移概率校准 + 期望回报联合优化 |
| 决策链 | 内部隐式多步前瞻，对外紧凑下一步 |
| 训练路线 | 小模型 + LoRA + 精简数据 |
| 基准 | 公开 + 自造结合 |
| 评测 | 校准误差（ECE/MCE）+ 回报误差 + 延迟 + 成本 |
| 流程 | 完整学术流程 |

---

## 三、系统架构设计（Proposed Changes）

### 3.1 输入 → 输出契约

**输入**：
```json
{
  "state": "用户问题/状态文本",
  "questions": {
    "q1": { "type": "choice", "options": ["选项1", "选项2", "选项3"], "instructions": "..." }
  }
}
```

**输出（紧凑下一步契约）**：
```json
{
  "answers": {
    "q1": {
      "type": "choice",
      "choice": "选项2",
      "prob_dist": { "选项1": {"结局A":0.1,"结局B":0.9}, "选项2": {"结局A":0.8,"结局B":0.2} },
      "expected_return": 0.73,
      "confidence": 0.91
    }
  }
}
```

### 3.2 总体架构图

```
用户 state + 选项(1-3)
        │
        ▼
┌───────────────────────────────┐
│  LLM 编码器 (开源小模型)          │  ← 语义理解
│  冻结 or LoRA 微调              │
└───────────────┬───────────────┘
                │ 语义嵌入 (潜空间)
                ▼
┌───────────────────────────────┐
│  MDP Head（隐式前瞻模块，新架构）  │  ← 核心创新
│  ┌─────────────────────────┐   │
│  │ 隐式多步前瞻展开           │   │  在潜空间内迭代式
│  │ (n步 Bellman 迭代操作)    │   │  展开决策链，不产生token
│  └────────────┬────────────┘   │
│               ▼                │
│  ┌─────────────────────────┐   │
│  │ 推断下一步最优决策          │   │  压回紧凑下一步
│  └─────────────────────────┘   │
└───────────────┬───────────────┘
                │ 动作价值 Q(s,a)
                ▼
┌───────────────────────────────┐
│  决策头族 (并行采样器)            │  ← 并行采样
│  - Choice头：选择概率            │
│  - Transition头：转移概率分布    │
│  - Return头：期望回报            │
│  - Confidence头：置信度          │
│  一次前向, 并行输出全部            │
└───────────────┬───────────────┘
                ▼
        紧凑结构化概率输出
```

### 3.3 MDP 统一框架设计

MDP 定义：`M = (S, A, P, R, γ)`

| 元素 | 定义 | 在模型中的表达 |
|------|------|---------------|
| **S**（状态） | 用户输入 state | LLM 编码器输出的语义嵌入 |
| **A**（动作） | 预定义选项 1-3 | Choice 头输出的动作选择 |
| **P**（转移） | P(s'|s,a) | **Transition 头**输出的转移概率分布 `prob_dist`（校准核心） |
| **R**（奖励） | 决策正确 + 转移准确 | 训练时的奖励函数 |
| **γ**（折扣） | 对未来回报折现 | MDP Head 内的 Bellman 迭代系数 |

### 3.4 隐式多步前瞻（核心理论模块）

**关键设计**：模型内部执行 `n` 步 Bellman 迭代（潜空间操作），计算每个动作的 Q 值：

```
Q_{k+1}(s,a) = R(s,a) + γ · Σ_{s'} P(s'|s,a) · max_{a'} Q_k(s', a')
```

- S 步迭代（如 4-8 步），全部在**潜空间**进行（用 MLP/注意力层近似 Bellman 更新）
- 最终取 `argmax_a Q_n(s,a)` 作为下一步最优决策
- 输出时压回紧凑契约（choice + prob_dist + expected_return + confidence）

**对外可见性**：用户只看"下一步最优决策 + 转移分布 + 回报"，多步前瞻是内部能力。

### 3.5 项目文件结构

```
Jev+MDP/
├── README.md
├── requirements.txt
├── pyproject.toml                  # 项目元数据
├── .trae/documents/                # 计划文档
├── config/
│   ├── base.yaml                   # 基础配置
│   └── train_rlcd_mdp.yaml         # 训练配置
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py              # LLM 编码器（冻结/LoRA）
│   │   ├── mdp_head.py             # ★ 隐式多步前瞻 MDP Head（核心创新）
│   │   ├── decision_heads.py       # Choice/Transition/Return/Confidence 头族
│   │   └── system_one_model.py     # 组装编码器+MDP Head+决策头
│   ├── sampler/
│   │   ├── __init__.py
│   │   └── parallel_sampler.py     # 并行采样器
│   ├── mdp/
│   │   ├── __init__.py
│   │   ├── mdp.py                  # MDP 定义与 Bellman 迭代
│   │   ├── reward.py               # 奖励函数（决策正确 + 转移准确）
│   │   └── transition.py           # 转移概率估计目标
│   ├── training/
│   │   ├── __init__.py
│   │   ├── rlcd_mdp_trainer.py     # ★ RLCD-MDP 训练器
│   │   └── losses.py               # 校准损失 + Q值损失
│   ├── data/
│   │   ├── __init__.py
│   │   ├── builder.py              # 数据构建
│   │   ├── external.py             # 公开数据集
│   │   └── synthetic.py            # 自造 MDP/决策链数据
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── calibration.py          # ECE/MCE 校准指标
│   │   ├── return_error.py         # 回报预测误差
│   │   ├── benchmark.py            # 决策基准评测
│   │   └── comparative.py          # 基线对比
│   └── utils/
│       ├── __init__.py
│       └── log.py                  # 日志工具
├── scripts/
│   ├── setup_env.ps1               # Windows 环境搭建
│   ├── download_data.py            # 数据下载
│   ├── train.py                    # 训练入口
│   ├── evaluate.py                 # 评测入口
│   └── generate_report.py          # 生成报告
├── tests/
│   ├── test_mdp.py
│   ├── test_mdp_head.py
│   ├── test_sampler.py
│   ├── test_model.py
│   └── test_calibration.py
└── paper/
    ├── outline.md                  # 论文大纲
    └── results/
```

---

## 四、实施步骤（分阶段，每阶段含验证）

### Phase 0：环境搭建
- [ ] 创建虚拟环境（`python -m venv .venv`）
- [ ] 安装依赖：`torch>=2.4`(CPU)、`transformers`、`datasets`、`PEFT`、`TRL`、`pytest`
- [ ] 验证 CPU 可加载最小基础模型（`distilbert-base` / `t5-small`）
- [ ] 编写 requirements.txt

### Phase 1：MDP 数学框架
- [ ] `mdp/mdp.py`：MDP 数据结构、Bellman 迭代算子 `Q_{k+1}`
- [ ] `mdp/reward.py`：奖励函数（决策正确 + 转移概率准确）
- [ ] `mdp/transition.py`：转移概率估计目标的形式化
- [ ] ✅ `pytest tests/test_mdp.py` — Bellman 收敛、MDP 逻辑正确
- **产出**：MDP 形式化文档（论文 Section 理论部分草稿）

### Phase 2：核心架构（编码器 + MDP Head + 决策头族）
- [ ] `models/encoder.py`：开源 LLM 编码器封装（冻结/LoRA）
- [ ] `models/mdp_head.py`：★ 隐式多步前瞻模块（潜空间 Bellman 迭代）
- [ ] `models/decision_heads.py`：Choice / Transition / Return / Confidence 头族
- [ ] `models/system_one_model.py`：组装完整模型
- [ ] `sampler/parallel_sampler.py`：并行采样器（一次前向输出全部）
- [ ] ✅ `pytest tests/test_mdp_head.py tests/test_model.py tests/test_sampler.py`

### Phase 3：数据构建
- [ ] `synthetic.py`：**首先构建自造 MDP 数据**（带明确真实转移概率，可验证校准）
  - 格式：state + 选项1-3 + 每个选项的真实转移分布 + 回报
- [ ] `external.py`：公开数据集（意图识别/工单分类等）转换为决策链格式
- [ ] 数据划分（train/val/test）

### Phase 4：RLCD-MDP 训练
- [ ] `training/losses.py`：
  - 校准损失（转移概率 vs 真实分布，如 KL/Brier/ECE-based）
  - Q 值损失（期望回报 vs 真实回报）
  - 决策损失（交叉熵）
- [ ] `training/rlcd_mdp_trainer.py`：联合训练循环
  1. 编码器 + MDP 前瞻前向
  2. 各头并行输出
  3. 计算校准 + Q + 决策损失
  4. 反向传播更新（含 LoRA）
- [ ] `scripts/train.py`：CPU 训练跑通
- [ ] ✅ `python scripts/train.py` — 损失下降，CPU 可运行

### Phase 5：评测与校准指标
- [ ] `eval/calibration.py`：ECE、MCE（Expected/Maximum Calibration Error）
- [ ] `eval/return_error.py`：回报预测误差（Q 值准确性）
- [ ] `eval/benchmark.py`：准确率、F1、校准、延迟、成本
- [ ] `eval/comparative.py`：对比基线（标准 softmax 分类头、无 MDP 前瞻）
- [ ] ✅ `python scripts/evaluate.py`

### Phase 6：消融实验 + 论文 + 报告
- [ ] 消融：有无 MDP Head、有无隐式前瞻、单步 vs 多步 Bellman 迭代
- [ ] 汇总实验表格
- [ ] `paper/outline.md`：完整论文大纲
- [ ] `scripts/generate_report.py`：实验报告
- [ ] 撰写论文正文（方法论、实验、分析）

---

## 五、边界场景与失败模式

### 边界场景
1. **选项少于动作空间**：固定决策头支持动态选项数量（1-3 个），padding 到 max
2. **转移分布不可知**：自造数据有真实分布；公开数据用模拟/标签推断近似转移
3. **多选项概率如何校准**：对每个选项单独预测转移到各结局的概率，再归一化

### 失败模式
| 情况 | 处理 |
|------|------|
| 模型过度自信（校准差） | 用 ECE 监测，调校准损失权重，必要时加温度缩放 |
| 隐式前瞻不收敛 | 固定步数（4-8）Bellman 迭代，检查 Q 值单调性 |
| CPU 训练过慢 | LoRA + 精简数据 + accumulate 步数，先跑通再扩展 |
| 公开数据无真实转移 | 用状态转移模拟器生成近似 ground-truth |

---

## 六、验证步骤（Verification）

### 每阶段验证命令
- **Phase 1**: `pytest tests/test_mdp.py`
- **Phase 2**: `pytest tests/test_mdp_head.py tests/test_model.py tests/test_sampler.py`
- **Phase 4**: `python scripts/train.py --config config/train_rlcd_mdp.yaml`
- **Phase 5**: `python scripts/evaluate.py --config config/train_rlcd_mdp.yaml`

### 最终验收标准
1. **决策能力**：对用户问题+选项1-3，输出紧凑下一步决策 + 转移概率分布 + 期望回报 + 置信度
2. **多步前瞻**：模型内部通过隐式 MDP Head 完成≥4步 Bellman 前瞻
3. **校准**：自造数据上 ECE/MCE 显著低于标准 softmax 基线（证明"校准=转移概率"落地）
4. **回报**：expected_return 预测误差低于随机基线
5. **可复现**：完整代码 + 实验报告 + 论文大纲

---

## 七、风险与缓解

| 风险 | 缓解 |
|------|------|
| Python 3.13 PyTorch 兼容 | PyTorch≥2.4，或用 conda py3.11 环境 |
| CPU 训练慢 | 小模型 (distilbert ~66M) + LoRA + 精简数据，目标内存可行 |
| "隐式多步前瞻"数学复杂 | 用 MLP 近似 Bellman 更新 + 消融实验证明其贡献 |
| 转移概率真实值难获取 | 自造 MDP 数据优先，用模拟器生成 ground-truth |
| 论文贡献不够聚焦 | 核心卖点锁定"隐式前瞻→紧凑输出"，MDP 统一校准+回报 |
