import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.log import Logger


def generate_report(experiments, output_dir="./paper/results"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    data = {
        "date": timestamp,
        "experiments": [],
    }

    for exp in experiments:
        row = {
            "name": exp.get("name", ""),
            "accuracy": exp.get("accuracy", 0),
            "ece": exp.get("ece", 0),
            "mce": exp.get("mce", 0),
            "mae": exp.get("mae", 0),
        }
        data["experiments"].append(row)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"report_{timestamp}.md"

    lines = [
        "# JEV+MDP 实验报告",
        "",
        f"- **生成时间**: {timestamp}",
        "",
        "## 实验结果对比",
        "",
        "| 实验 | 准确率 | ECE | MCE | MAE |",
        "|------|--------|-----|-----|-----|",
    ]

    for exp in data["experiments"]:
        lines.append(
            f"| {exp['name']} | {exp['accuracy']:.4f} | {exp['ece']:.4f} | "
            f"{exp['mce']:.4f} | {exp['mae']:.4f} |"
        )

    content = "\n".join(lines)
    output_file.write_text(content, encoding="utf-8")
    print(f"报告已保存: {output_file}")


if __name__ == "__main__":
    experiments = [
        {
            "name": "基线 (标准Softmax)",
            "accuracy": 0.5,
            "ece": 0.2,
            "mce": 0.3,
            "mae": 1.5,
        }
    ]
    generate_report(experiments)
