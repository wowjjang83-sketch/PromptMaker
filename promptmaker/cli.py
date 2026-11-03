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
    p.add_argument("--check", action="store_true", help="요소 태그/관계 규칙 점검 (내가 추가한 줄 확인용)")
    args = p.parse_args(argv)

    maker = PromptMaker(user_data_dir=args.user_data)
    if args.list:
        for t in maker.templates.values():
            print(f"[{t.type}] {t.id} - {t.name}")
        print()
        for k, e in sorted(maker.elements.items()):
            print(f"{k}: {len(e.items)}개")
        return 0

    if args.check:
        return check(maker)

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


def check(maker: PromptMaker) -> int:
    """Lint the element files: unknown tags and requirements nothing can provide."""
    rel = maker.relations
    known = set(rel.tag_words) | set(rel.prop_words) | set(rel.implies) | set(rel.affinity)
    for vals in rel.implies.values():
        known |= set(vals)
    for a, b in rel.conflicts:
        known |= {a, b}
    for aff in rel.affinity.values():
        known |= set(aff)
    provided = set()
    for key in list(rel.providers) + list(rel.garments) + ["outfit_type"]:
        for it in maker.elements.get(key, []).items if key in maker.elements else []:
            provided |= rel.closure(it.tags)
    provided |= {i.text for i in maker.elements.get("outfit_type").items} if "outfit_type" in maker.elements else set()
    problems = 0
    for key, el in sorted(maker.elements.items()):
        tagged = sum(1 for i in el.items if i.tags or i.requires or i.excludes)
        print(f"{key}: {len(el.items)}개 항목, 태그 있는 항목 {tagged}개")
        for it in el.items:
            unknown = (it.tags | it.requires | it.excludes) - known - provided
            if unknown:
                problems += 1
                print(f"  ? '{it.text}': 규칙에 없는 태그 {sorted(unknown)} (relations.json 에 추가하면 효과가 생깁니다)")
            never = it.requires - provided
            if never:
                problems += 1
                print(f"  ! '{it.text}': +{'/'.join(sorted(never))} 를 제공하는 장소/의상 항목이 없어 장면이 정해지면 선택되지 않습니다")
    print(f"\n점검 완료: 확인할 항목 {problems}개")
    return 0


if __name__ == "__main__":
    sys.exit(main())
