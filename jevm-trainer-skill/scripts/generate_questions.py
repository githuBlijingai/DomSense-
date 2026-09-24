"""Generate JEV+MDP decision scenarios (出题器).

Two modes:
  1. --file, --scenarios path  : load hand-authored scenarios from a JSON file.
  2. default                   : emit a deterministic built-in template set
                                 so the pipeline is runnable even without a
                                 hand-authored file (useful for smoke tests).

Each scenario is a dict:
  state_text, actions, optimal_action, transition_probs, expected_returns, domain
"""

import argparse
import json
import random
import sys


def _pack(state_text: str, actions: list[str], optimal_action: int,
          success_probs: list[float], domain: str) -> dict:
    """Build a well-formed Scenario from parts.

    transition_probs[i] = [p_i, 1-p_i] (二元 outcome).
    expected_returns[i] = p_i.
    """
    transition_probs = [[p, round(1.0 - p, 4)] for p in success_probs]
    expected_returns = [round(p, 4) for p in success_probs]
    return {
        "state_text": state_text,
        "actions": actions,
        "optimal_action": optimal_action,
        "transition_probs": transition_probs,
        "expected_returns": expected_returns,
        "domain": domain,
    }


def builtin_scenarios() -> list[dict]:
    """A deterministic, domain-diverse set used as a fallback / smoke test."""
    return [
        _pack(
            "一位55岁患者同时患有高血压和轻度肾功能不全，需选择降压药。氨氯地平降压稳妥但对肾保护一般；"
            "缬沙坦对肾功能有保护但可能升高肌酐；双氢克尿噻价廉但长期可能影响电解质。",
            ["氨氯地平", "缬沙坦", "双氢克尿噻"],
            1,
            [0.62, 0.80, 0.55],
            "medical",
        ),
        _pack(
            "你有10万元闲置资金，风险偏好中等。国债年化3.1%几乎无风险；沪深300指数基金历史年化约8%但回撤可达20%；"
            "混合债基年化5%波动小。请分配资金进行长期投资。",
            ["全部买入国债", "全部买入沪深300指数基金", "买入混合债基"],
            2,
            [0.55, 0.70, 0.78],
            "finance",
        ),
        _pack(
            "一个初中生基础薄弱，在数学解题上屡屡受挫，需要在在两周内提升。选项：高强度刷题冲刺；"
            "拆分小目标逐级巩固基础；报集训营接受统一强度训练。",
            ["高强度刷题冲刺", "拆分小目标逐级巩固", "报集训营统一强度"],
            1,
            [0.50, 0.82, 0.58],
            "education",
        ),
        _pack(
            "客服系统同时涌入大量工单：一个VIP客户的账户无法登录（高价值）、一个普通客户的Payment已扣款但未到账（时间敏感）、"
            "一个批量邮件营销用户咨询（低紧急）。只能优先处理部分。",
            ["优先VIP登录问题", "优先Payment到账问题", "优先批量咨询"],
            1,
            [0.66, 0.79, 0.40],
            "customer_support",
        ),
        _pack(
            "医疗资源紧张需决定肾透析设备优先分配：一位年轻父亲（预后好），一位老年学者（贡献大），一位有希望的年轻患者（治愈率高）",
            ["优先年轻父亲", "优先老年学者", "优先年轻治愈患者"],
            1,
            [0.60, 0.62, 0.72],
            "ethical",
        ),
    ]


def gen_count(base: list[dict], count: int, seed: int | None = None) -> list[dict]:
    """If count <= len(base) return first count; else sample with replacement."""
    if seed is not None:
        random.seed(seed)
    if count <= len(base):
        return base[:count]
    return [random.choice(base) for _ in range(count)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", "--scenarios", dest="scenarios")
    parser.add_argument("--domain", help="Filter built-in scenarios by domain.")
    parser.add_argument("--count", default=5, type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", help="Write scenario list to this JSON file.")
    parser.add_argument("--json", dest="json_out", action="store_true",
                        help="Print scenarios as raw JSON array instead of args summary.")
    args = parser.parse_args(argv)

    if args.scenarios:
        with open(args.scenarios, "r", encoding="utf-8") as fh:
            scenarios = json.load(fh)
    else:
        scenarios = builtin_scenarios()
        if args.domain:
            scenarios = [s for s in scenarios if s.get("domain") == args.domain]

    selected = gen_count(scenarios, args.count, args.seed)
    if args.json_out:
        print(json.dumps(selected, ensure_ascii=False, indent=2))
    elif args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(selected, fh, ensure_ascii=False, indent=2)
        print(json.dumps({"out": args.out, "scenarios": len(selected)}, ensure_ascii=False))
    else:
        print(json.dumps(
            {"scenarios": len(selected),
             "domains": sorted({s.get("domain") for s in selected})},
            ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
