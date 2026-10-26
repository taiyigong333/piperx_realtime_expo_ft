"""生成上游固定路径链接；链接目标仍是主仓库管理的 OpenPI 子模块。"""

import os
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
link = root / "third_party/expo/expo_ft/agents/vla/openpi"
target = root / "third_party/openpi"
if not (target / "pyproject.toml").is_file():
    raise SystemExit("请先初始化 third_party/openpi 子模块")
if link.is_symlink():
    if link.resolve() != target.resolve():
        raise SystemExit(f"现有链接指向不同目录：{link}")
elif link.exists():
    raise SystemExit(f"目标已有独立目录，请人工确认：{link}")
else:
    link.symlink_to(os.path.relpath(target, link.parent), target_is_directory=True)
# 生成链接属于本机运行配置；用子模块本地 exclude 保持上游工作树干净。
exclude = Path(
    subprocess.check_output(
        ["git", "-C", str(root / "third_party/expo"), "rev-parse", "--git-path", "info/exclude"], text=True
    ).strip()
)
exclude.parent.mkdir(parents=True, exist_ok=True)
existing = exclude.read_text() if exclude.exists() else ""
pattern = "/expo_ft/agents/vla/openpi"
if pattern not in existing.splitlines():
    exclude.write_text(existing + "\n" + pattern + "\n")
print(f"OpenPI 路径就绪：{link} -> {target}")
