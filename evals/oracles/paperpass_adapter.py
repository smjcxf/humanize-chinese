#!/usr/bin/env python3
"""
PaperPass AIGC detection oracle.

Headless-browser adapter that submits text to PaperPass free AIGC check
(5/day quota) and parses the score. Uses a persistent Chrome profile so
the user logs in once and cookies stay valid.

Architecture:
  This Python module shells out to a Node Playwright script
  (scripts/browser/paperpass-check.js) which:
    1. Loads /panel/index to grab CSRF token
    2. POSTs to /panel/index/submit-papers-key with text (skipping paid services)
    3. Polls /panel/report for the newly-submitted entry reaching 已完成
    4. Parses AIGC 总体疑似度 percentage
    5. Returns JSON: {"score": int, "similarity": int, "status": "...", "error": ...}

Usage (from Python):
    from evals.oracles.paperpass_adapter import check
    result = check("中文文本内容...")
    print(result["score"])  # 0-100 AIGC score
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # .../humanize-chinese/
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(REPO_ROOT))  # .../humanize/

PLAYWRIGHT_SCRIPT = os.path.join(
    WORKSPACE_ROOT, 'scripts', 'browser', 'paperpass-check.js'
)
BROWSER_LOCK = os.path.join(WORKSPACE_ROOT, 'scripts', 'browser-lock.sh')


def check(text: str, timeout_seconds: int = 180) -> dict:
    """Submit text to PaperPass and return AIGC score.

    Args:
        text: Chinese text (300-150000 chars)
        timeout_seconds: max wait for report

    Returns:
        dict with keys:
          - score: int 0-100 AIGC detection score, or None on error
          - similarity: int 0-100 plagiarism similarity, or None
          - status: "ok" | "error" | "timeout" | "quota_exceeded"
          - error: error message if any
          - elapsed_seconds: float
    """
    if not os.path.exists(PLAYWRIGHT_SCRIPT):
        return {'score': None, 'status': 'error',
                'error': f'Script missing: {PLAYWRIGHT_SCRIPT}'}
    if not os.path.exists(BROWSER_LOCK):
        return {'score': None, 'status': 'error',
                'error': f'Lock script missing: {BROWSER_LOCK}'}

    # Quick sanity check — PaperPass requires 300+ Chinese chars
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    if chinese_chars < 300:
        return {'score': None, 'status': 'error',
                'error': f'Too short: {chinese_chars} Chinese chars (min 300)'}
    if chinese_chars > 150000:
        return {'score': None, 'status': 'error',
                'error': f'Too long: {chinese_chars} Chinese chars (max 150000)'}

    # Write text to a temp file; pass path as script arg. Stdin piping through
    # browser-lock.sh's `node "$@" &` can eat the input, file arg is reliable.
    import tempfile
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
                'error': 'playwright script timeout',
                'elapsed_seconds': time.time() - start}
    finally:
        try: os.unlink(tmp_path)
        except OSError: pass

    elapsed = time.time() - start
    # Parse last JSON line from stdout
    for line in reversed(proc.stdout.strip().split('\n')):
        line = line.strip()
        if not line.startswith('{'):
            continue
        try:
            obj = json.loads(line)
            obj['elapsed_seconds'] = round(elapsed, 1)
            return obj
        except json.JSONDecodeError:
            continue

    return {
        'score': None, 'status': 'error',
        'error': f'No JSON in output. exit={proc.returncode} stderr={proc.stderr[-500:]}',
        'elapsed_seconds': round(elapsed, 1),
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PaperPass AIGC oracle adapter')
    parser.add_argument('file', nargs='?', help='Text file (default: stdin)')
    parser.add_argument('--timeout', type=int, default=180)
    args = parser.parse_args()

    text = open(args.file, encoding='utf-8').read() if args.file else sys.stdin.read()
    result = check(text, timeout_seconds=args.timeout)
    print(json.dumps(result, ensure_ascii=False, indent=2))
