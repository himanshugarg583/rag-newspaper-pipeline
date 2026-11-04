"""Response helpers for consistent API output."""

import math


def total_pages(total: int, page_size: int) -> int:
    if total <= 0:
        return 0
    return math.ceil(total / page_size)
