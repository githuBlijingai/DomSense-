"""诊断验证：两个关键局限是否已修复。

局限 1: 动作数固定为 3 → 现在支持任意动作数（2/3/4...）
局限 2: 位置顺序隐含信息 → 交叉注意力模式下重排选项位置应语义等变
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import yaml

from src.models.system_one_model import SystemOneModel


def load_config():
    with open("config/base.yaml", "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    return base["model"]


def build_model(checkpoint=None):
    config = load_config()
    model = SystemOneModel(config)
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def check_variable_actions(model):
    """局限 1: 验证 2、3、4 个动作都正确工作，输出维度随之变化。"""
    print("\n=== 局限 1: 动作数可变 ===")
    texts = ["今天有点累，想放松一下"]
    ok = True
    for n, actions in [
        (2, ["去跑步", "在家睡觉"]),
        (3, ["去跑步", "去书店", "看电影"]),
        (4, ["去跑步", "去书店", "看电影", "吃大餐"]),
    ]:
        with torch.no_grad():
            r = model(texts, action_texts=[actions], num_valid_actions=torch.tensor([n]))
        probs = r["choice_probs"][0].numpy()
        choice = int(r["choices"][0].item())
        expected_returns = r["expected_returns"][0].numpy()
        status = "OK" if choice < n and probs.shape[0] == n else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  {n} 个动作: shape={probs.shape}, choice={choice}, "
              f"returns_shape={expected_returns.shape} -> {status}")
    print(f"  局限 1 结果: {'通过' if ok else '失败'}")
    return ok


def check_position_invariance(model):
    """局限 2: 交叉注意力模式下重排选项位置应给出语义等变的概率分布。"""
    print("\n=== 局限 2: 位置顺序隐含信息 ===")
    texts = ["今天有点累，想放松一下"]
    actions = ["去散步", "去吃火锅", "去看电影"]
    perm = [2, 0, 1]
    actions_b = [actions[p] for p in perm]

    with torch.no_grad():
        ra = model(texts, action_texts=[actions], num_valid_actions=torch.tensor([3]))
        rb = model(texts, action_texts=[actions_b], num_valid_actions=torch.tensor([3]))

    probs_a = ra["choice_probs"][0].numpy()
    probs_b = rb["choice_probs"][0].numpy()
    permuted_a = probs_a[perm]
    max_diff = float(np.abs(probs_b - permuted_a).max())
    print(f"  原排列 probs_a: {probs_a.round(4)}")
    print(f"  重排后 probs_b: {probs_b.round(4)}")
    print(f"  语义重排 probs_a[perm]: {permuted_a.round(4)}")
    print(f"  最大差异 max_diff: {max_diff:.4f}")
    # 若位置不再隐含信息，重排后概率应接近语义等变（max_diff 应远小于未训练噪音）
    ok = max_diff < 0.3
    print(f"  局限 2 结果: {'通过' if ok else '失败'}")
    return ok


def main():
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else None
    model = build_model(checkpoint)

    ok1 = check_variable_actions(model)
    ok2 = check_position_invariance(model)
    print("\n" + "=" * 40)
    print(f"总体: 局限1({'通过' if ok1 else '失败'}), 局限2({'通过' if ok2 else '失败'})")
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    sys.exit(main())
