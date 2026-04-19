#!/usr/bin/env python3
"""
朱雀 (Zhuque, 腾讯) AI text detection oracle.

Free, no login, 5/day quota. Headless browser adapter.

Output:
    {
      "score": int,          # 0-100 AI likelihood = ai_feature + suspect_ai
      "ai_feature": float,   # AI特征 %
      "suspect_ai": float,   # 疑似AI %
      "human_feature": float,# 人工特征 %
      "status": "ok" | "error" | "timeout" | "quota_exceeded",
      "error": str,
      "remaining_quota": int,
      "elapsed_seconds": float
    }

Usage (from Python):
    from evals.oracles.zhuque_adapter import check
    result = check("中文文本...")
    print(result["score"])
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(REPO_ROOT))

PLAYWRIGHT_SCRIPT = os.path.join(
    WORKSPACE_ROOT, 'scripts', 'browser', 'zhuque-check.js'
)
BROWSER_LOCK = os.path.join(WORKSPACE_ROOT, 'scripts', 'browser-lock.sh')


def check(text: str, timeout_seconds: int = 90) -> dict:
    """Submit text to 朱雀 and return AI detection score."""
    if not os.path.exists(PLAYWRIGHT_SCRIPT):
        return {'score': None, 'status': 'error',
                'error': f'Script missing: {PLAYWRIGHT_SCRIPT}'}

    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    if chinese_chars < 100:
        return {'score': None, 'status': 'error',
                'error': f'Too short: {chinese_chars} Chinese chars'}

    with tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
        tmp.write(text)
        tmp_path = tmp.name

    start = time.time()
    try:
        proc = subprocess.run(
            [BROWSER_LOCK, 'run', '--timeout', str(timeout_seconds),
             PLAYWRIGHT_SCRIPT, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds + 30,
        )
    except subprocess.TimeoutExpired:
        return {'score': None, 'status': 'timeout',
                'error': 'browser script timeout',
                'elapsed_seconds': round(time.time() - start, 1)}
    finally:
        try: os.unlink(tmp_path)
        except OSError: pass

    elapsed = round(time.time() - start, 1)
    for line in reversed(proc.stdout.strip().split('\n')):
        line = line.strip()
        if not line.startswith('{'):
            continue
        try:
            obj = json.loads(line)
            obj['elapsed_seconds'] = elapsed
            return obj
        except json.JSONDecodeError:
            continue

    return {
        'score': None, 'status': 'error',
        'error': f'No JSON in output. exit={proc.returncode}',
        'stderr_tail': proc.stderr[-400:],
        'elapsed_seconds': elapsed,
    }


if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='Zhuque AI detection oracle')
    parser.add_argument('file', nargs='?', help='Text file (default: stdin)')
    parser.add_argument('--timeout', type=int, default=90)
    args = parser.parse_args()
    text = open(args.file, encoding='utf-8').read() if args.file else sys.stdin.read()
    result = check(text, timeout_seconds=args.timeout)
    print(json.dumps(result, ensure_ascii=False, indent=2))
