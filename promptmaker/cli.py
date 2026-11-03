"""Command line interface.

    python -m promptmaker "비 오는 날 카페" --type image --count 3
"""
from __future__ import annotations

import argparse
import json
import sys

from .core import MODES, GenerateOptions, PromptMaker
from .data_loader import RATINGS
from .safety import SafetyError


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="promptmaker", description="LLM 없이 영문 자연어 프롬프트 생성")
    p.add_argument("text", nargs="?", default="", help="사용자 요청 (한글/영문)")
    p.add_argument("--type", dest="prompt_type", default="image", choices=["image", "video"])
    p.add_argument("--template", default=None, help="템플릿 id (또는 random)")
    p.add_argument("--mode", default="random_by_type", choices=MODES)
    p.add_argument("--rating", dest="content_rating", default="sfw", choices=RATINGS)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="요소 고정값 (여러 번 사용 가능) 예: --set nationality=Korean")
    p.add_argument("--random", dest="random_elements", default="",
                   help="요소별 랜덤에서 랜덤으로 뽑을 요소 (쉼표 구분)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--no-translate", action="store_true")
    p.add_argument("--user-data", default=None)
    p.add_argument("--json", action="store_true", help="상세 결과를 JSON 으로 출력")
    p.add_argument("--list", action="store_true", help="템플릿/요소 목록 출력")
    args = p.parse_args(argv)

    maker = PromptMaker(user_data_dir=args.user_data)
    if args.list:
        for t in maker.templates.values():
            print(f"[{t.type}] {t.id} - {t.name}")
        print()
        for k, e in sorted(maker.elements.items()):
            print(f"{k}: {len(e.items)}개")
        return 0

    selections = dict(s.split("=", 1) for s in args.set if "=" in s)
    opts = GenerateOptions(
        prompt_type=args.prompt_type, template=args.template, mode=args.mode,
        content_rating=args.content_rating, selections=selections,
        random_elements=[s.strip() for s in args.random_elements.split(",") if s.strip()],
        seed=args.seed, translate=not args.no_translate,
    )
    try:
        results = maker.generate_batch(args.text, args.count, opts)
    except SafetyError as exc:
        print(exc, file=sys.stderr)
        return 2
    for r in results:
        if args.json:
            print(json.dumps(r.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(r.prompt)
            for w in r.warnings:
                print(f"  ! {w}", file=sys.stderr)
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
