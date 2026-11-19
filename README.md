# PromptMaker

LLM 없이 **규칙 + 템플릿 + 확률 기반 단어 목록**으로 영문 자연어 프롬프트(이미지 생성용 / 영상 생성용)를 만드는 파이썬 라이브러리입니다.
파이썬 앱, ComfyUI 커스텀 노드, HTML 테스트 페이지에서 같은 엔진을 사용합니다.

> 모든 인물은 **20세 이상 성인**으로만 생성됩니다. 미성년자를 연상시키는 입력은 거부되고, 20 미만 나이는 20으로 보정됩니다.

## 빠른 시작

```bash
pip install -e .            # 또는 저장소 폴더에서 바로 사용 (외부 패키지 불필요)
python -m promptmaker "비 오는 날 카페 창가의 여자" --type image --count 3
python -m promptmaker.server   # http://127.0.0.1:8765 HTML 테스트 페이지
```

```python
from promptmaker import PromptMaker, GenerateOptions

pm = PromptMaker()
result = pm.generate("해변 노을", GenerateOptions(prompt_type="image", mode="random_by_type"))
print(result.prompt)
```

## 처리 과정

옵션 선택 → 사용자 입력 → (영어가 아니면) 구글 번역 → 입력 단어 검색 → 확률 기반 관련 단어(형용사/동사/명사) 추출 → 템플릿에 영문 문법에 맞게 채우기 → 결과 출력

## 구성

| 경로 | 내용 |
|---|---|
| `promptmaker/` | 라이브러리 본체 (엔진, 문법, 번역, 키워드 검색, 안전 규칙) |
| `promptmaker/data/templates/*.json` | 템플릿 10종 (이미지 7, 영상 3) — 사용자 추가/수정 가능 |
| `promptmaker/data/elements/*.txt, *.json` | 템플릿 요소 목록 (30여 종) — 사용자 추가/수정 가능 |
| `promptmaker/data/packs/wildcard/elements/*.txt` | 와일드카드 확장 요소 약 2만 1천 줄 (기본은 꺼짐, `PromptMaker(packs=["wildcard"])`, 0.12.0, 가이드 7-7) |
| `promptmaker/data/lexicon/` | 관련 단어 사전, 오프라인 한영 사전 |
| `promptmaker/data/system/`, `model_examples/` | 고정 입력 (시스템 프롬프트, 모델별 예시) — 추후 LLM 옵션용 |
| `promptmaker/web/index.html` | HTML 테스트 페이지 |
| `promptmaker/body_parts.py` | 신체 부위별 클로즈업(다리·입술·손·발 …)과 이미지 분석 구도 문구 분리 (0.11.0, 가이드 7-6) |
| `data/vision/outfit_prompt.txt`, `vision.KEY_HINTS` | 이미지 분석에서 의상만 한 번 더 묻기 (겹쳐 입은 코르셋·벨트 등 빠짐 방지), 의상 상태에 들어간 의상 이름은 해당 의상 요소로 (0.15.0, 가이드 7-1) |
| `promptmaker/distance.py`, `data/elements/distance.txt`, `promptmaker/aspect.py` | 피사체 거리(가까이서·중거리·멀리서·원경)와 구도 연동, 프롬프트로 화면 비율 추천 `suggest_aspect` (0.14.0, 가이드 7-9) |
| `promptmaker/masturbation.py`, `data/elements/masturbation.txt` | 자위 장면 (가슴 주무르기 · 클리토리스 만지기 · 질에 손가락 넣기) 과 요청 내용 반영 강화 (0.13.0, 가이드 7-8) |
| `promptmaker/framing.py`, `undressing.py` | 샷 구도(얼빡샷·클로즈업·미디엄·카우보이·전신)와 탈의 장면 규칙 — 구도·동작·의상이 서로 맞게 (0.9.0, 가이드 7-5) |
| `promptmaker/element_check.py` | 요소 검사: 문장 조각 거르기 · 다듬기 · 잘못된 요소 파일의 줄을 맞는 요소로 (0.8.0 · 0.10.0 보강, 가이드 7-4) |
| `promptmaker/vision.py` | (선택) 이미지 분석 → 요소 txt 자동 추가 (기본 모델 Qwen3-VL-4B, 교체 가능) |
| `comfyui_nodes.py`, `__init__.py` | ComfyUI 커스텀 노드 |
| `tools/github_auto.py` | 토큰 파일 기반 GitHub 자동 관리 |
| `docs/GUIDE.md` | **사용 가이드 (필독)** |
| `docs/REFERENCES.md` | 참고한 유사 프로젝트 |

## 테스트

```bash
python -m unittest discover -s tests -t .
```
