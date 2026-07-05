from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import embedding_info, rebuild_vector_index


def main() -> int:
    count = rebuild_vector_index()
    result = {
        "ok": True,
        "indexed_chunks": count,
        "embedding": embedding_info(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["embedding"] and not result["embedding"].get("model_ready"):
        print(
            "提示：当前仍在使用 fallback embedding。安装 requirements-embedding.txt 后重新运行，可切换到真实中文模型。",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
