#!/usr/bin/env python3
"""One-turn tool selection regression kit. Standard library only. No tools are executed."""
import argparse
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {key: {"type": "string"}},
                "required": [key],
                "additionalProperties": False,
            },
        },
    }
    for name, description, key in [
        ("get_item", "按商品编号查询商品详情", "item_id"),
        ("get_stock", "按商品 SKU 查询当前库存", "sku"),
        ("search_help", "检索商城帮助文档", "query"),
        ("add_to_wishlist", "按 SKU 加入心愿清单（此示例只评测调用意图，不执行）", "sku"),
    ]
]

SYSTEM_PROMPT = (
    "你是虚构商品目录的助理。只能在用户明确请求且提供必需编号时选择相应工具。"
    "工具无法完成的任务、普通闲聊、无需查数据的问题，以及缺少必要编号的请求，请直接回答或追问，"
    "不要猜测编号或调用不相关工具。"
)


def load_rows(path):
    seen = set()
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                identifier = row["id"]
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{lineno}: invalid row: {exc}") from exc
            if identifier in seen:
                raise ValueError(f"{path}:{lineno}: duplicate id {identifier}")
            seen.add(identifier)
            rows.append(row)
    return rows


def normalize_calls(raw_calls):
    calls = []
    for call in raw_calls or []:
        func = call.get("function", call)
        args = func.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
        if not isinstance(args, dict):
            raise ValueError("tool arguments must be a JSON object")
        calls.append({"name": func["name"], "arguments": args})
    return calls


def score(cases_path, pred_path):
    cases = load_rows(cases_path)
    preds = {row["id"]: row for row in load_rows(pred_path)}
    passes = 0
    called_passes = called_total = no_call_passes = no_call_total = 0
    for case in cases:
        expected = normalize_calls(case["expected"])
        pred = preds.get(case["id"])
        error = None
        try:
            if pred is None:
                raise ValueError("prediction missing")
            if pred.get("error"):
                raise ValueError(str(pred["error"]))
            actual = normalize_calls(pred.get("tool_calls", []))
            match = actual == expected
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            error = str(exc)
            actual = []
            match = False
        if expected:
            called_total += 1
            called_passes += match
        else:
            no_call_total += 1
            no_call_passes += match
        passes += match
        if not match:
            print(f"FAIL {case['id']}: expected={expected!r} actual={actual!r}" + (f" error={error}" if error else ""))
    extras = sorted(set(preds) - {case["id"] for case in cases})
    if extras:
        print(f"IGNORED extra ids: {extras}")
    print(f"Exact match: {passes}/{len(cases)}")
    print(f"Should call: {called_passes}/{called_total}; should not call: {no_call_passes}/{no_call_total}")
    return passes == len(cases)


def run(cases_path, output_path, model, url, timeout):
    cases = load_rows(cases_path)
    with Path(output_path).open("w", encoding="utf-8") as stream:
        for case in cases:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": case["user"]},
                ],
                "tools": TOOLS,
                "tool_choice": "auto",
                "temperature": 0,
            }
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            request = Request(
                url,
                data=data,
                headers={"Content-Type": "application/json", "Authorization": "Bearer local-test"},
                method="POST",
            )
            row = {"id": case["id"]}
            try:
                with urlopen(request, timeout=timeout) as response:
                    result = json.load(response)
                message = result["choices"][0]["message"]
                row["tool_calls"] = normalize_calls(message.get("tool_calls", []))
            except (Exception,) as exc:  # Report per-case failures rather than silently scoring them as no-call.
                row["error"] = f"{type(exc).__name__}: {exc}"
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{row['id']}: {'error' if 'error' in row else 'captured'}")


def selftest():
    assert normalize_calls([{"function": {"name": "get_stock", "arguments": '{"sku":"SKU-204"}'}}]) == [
        {"name": "get_stock", "arguments": {"sku": "SKU-204"}}
    ]
    assert normalize_calls([]) == []
    try:
        normalize_calls([{"function": {"name": "get_stock", "arguments": "{bad}"}}])
    except ValueError:
        pass
    else:
        raise AssertionError("Malformed tool JSON should fail")
    print("Parser self-test passed (no model was run).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="Query a local OpenAI-compatible chat completions server")
    run_parser.add_argument("--cases", required=True)
    run_parser.add_argument("--out", required=True)
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    run_parser.add_argument("--timeout", type=int, default=60)
    score_parser = sub.add_parser("score", help="Score existing prediction JSONL without model access")
    score_parser.add_argument("--cases", required=True)
    score_parser.add_argument("--pred", required=True)
    sub.add_parser("selftest", help="Check parser behavior without accessing any model")
    args = parser.parse_args()
    if args.command == "selftest":
        selftest()
    elif args.command == "run":
        run(args.cases, args.out, args.model, args.url, args.timeout)
    elif not score(args.cases, args.pred):
        sys.exit(1)


if __name__ == "__main__":
    main()
