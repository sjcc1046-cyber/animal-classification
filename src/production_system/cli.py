"""Command-line interface for the production-system demonstration."""

from __future__ import annotations

import argparse
import sys

from .knowledge_base import EXAMPLES, FEATURES, create_engine


def _print_result(facts: set[str] | frozenset[str]) -> int:
    result = create_engine().infer(facts)
    print(f"初始事实：{', '.join(sorted(result.initial_facts)) or '无'}")
    print("推理过程：")
    if not result.fired_rules:
        print("  没有规则被触发")
    for step, fired in enumerate(result.fired_rules, start=1):
        rule = fired.rule
        premises = " AND ".join(sorted(rule.premises))
        print(f"  {step}. {rule.rule_id}: IF {premises} THEN {rule.conclusion}")
        if rule.explanation:
            print(f"     {rule.explanation}")

    if result.status == "recognized":
        print(f"识别结果：{result.animal}")
        return 0
    if result.status == "ambiguous":
        print(f"识别结果：存在冲突，候选为 {', '.join(result.candidates)}")
        return 2
    print("识别结果：信息不足或不在知识库范围内")
    return 1


def _interactive() -> int:
    names = list(FEATURES)
    print("请选择观察到的动物特征（用空格分隔编号）：")
    for index, name in enumerate(names, start=1):
        print(f"  {index:>2}. {name}：{FEATURES[name]}")
    raw = input("> ").strip()
    try:
        indexes = {int(item) for item in raw.split()}
    except ValueError:
        print("输入无效：请输入用空格分隔的数字编号")
        return 2
    invalid = sorted(index for index in indexes if not 1 <= index <= len(names))
    if invalid:
        print(f"输入无效：不存在编号 {invalid}")
        return 2
    return _print_result({names[index - 1] for index in indexes})


def main() -> int:
    # Keep Chinese inference traces readable in Windows terminals and redirected logs.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="六类动物产生式识别系统")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--example", choices=EXAMPLES, help="运行指定动物的标准案例")
    group.add_argument("--facts", nargs="+", help="直接提供中文事实，例如：有毛发 吃肉 家养 会喵叫")
    group.add_argument("--list-features", action="store_true", help="列出可输入特征")
    args = parser.parse_args()

    if args.list_features:
        for name, description in FEATURES.items():
            print(f"{name}: {description}")
        return 0
    if args.example:
        return _print_result(EXAMPLES[args.example])
    if args.facts:
        unknown = sorted(set(args.facts) - FEATURES.keys())
        if unknown:
            parser.error(f"未知事实：{', '.join(unknown)}")
        return _print_result(set(args.facts))
    return _interactive()


if __name__ == "__main__":
    raise SystemExit(main())
