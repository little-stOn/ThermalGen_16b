#!/usr/bin/env python3
"""Keep a requested CUDA device active during long data-preparation work."""

from __future__ import annotations

import argparse
import signal
import time

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--matrix-size", type=int, default=2048)
    parser.add_argument("--duty-cycle", type=float, default=0.35)
    parser.add_argument("--period", type=float, default=0.50)
    args = parser.parse_args()
    if args.matrix_size < 256 or not 0.0 < args.duty_cycle <= 1.0 or args.period <= 0.0:
        raise SystemExit("invalid keepalive parameters")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("keepalive requires a CUDA device")
    torch.cuda.set_device(device)

    left = torch.randn(
        (args.matrix_size, args.matrix_size), device=device, dtype=torch.float16
    )
    right = torch.randn_like(left)
    output = torch.empty_like(left)
    stopping = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    with torch.inference_mode():
        while not stopping:
            start = time.monotonic()
            deadline = start + args.period * args.duty_cycle
            while time.monotonic() < deadline and not stopping:
                torch.mm(left, right, out=output)
                torch.cuda.synchronize(device)
            remaining = args.period - (time.monotonic() - start)
            if remaining > 0.0 and not stopping:
                time.sleep(remaining)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# 说明：该脚本不是数据集 loader，而是长时间索引/测试时的 CUDA 保活器。
# 它在指定 device 上循环固定输入的矩阵乘法，并在每次运算后同步，按
# duty-cycle/period 控制负载；固定输入和复用 output 可避免长期运行时的
# 数值溢出、异步队列积压和额外显存分配。需要结束时发送 SIGTERM/SIGINT。
