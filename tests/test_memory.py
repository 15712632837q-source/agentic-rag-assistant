"""记忆模块的纯函数：余弦相似度 + 时间衰减（不依赖模型/DB）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import memory  # noqa: E402


def test_cosine():
    assert memory._cos([1, 0], [1, 0]) == 1.0
    assert memory._cos([1, 0], [0, 1]) == 0.0
    assert round(memory._cos([1, 1], [1, 0]), 3) == 0.707
    assert memory._cos([0, 0], [1, 1]) == 0.0   # 零向量不报错


def test_time_decay():
    assert memory._time_decay(0) == 1.0
    assert memory._time_decay(memory.HALF_LIFE_DAYS) == 0.5
    assert memory._time_decay(2 * memory.HALF_LIFE_DAYS) == 0.25
    # 越旧权重越低
    assert memory._time_decay(1) > memory._time_decay(30)
