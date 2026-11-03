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
| `promptmaker/data/lexicon/` | 관련 단어 사전, 오프라인 한영 사전 |
| `promptmaker/data/system/`, `model_examples/` | 고정 입력 (시스템 프롬프트, 모델별 예시) — 추후 LLM 옵션용 |
| `promptmaker/web/index.html` | HTML 테스트 페이지 |
| `comfyui_nodes.py`, `__init__.py` | ComfyUI 커스텀 노드 |
| `tools/github_auto.py` | 토큰 파일 기반 GitHub 자동 관리 |
| `docs/GUIDE.md` | **사용 가이드 (필독)** |
| `docs/REFERENCES.md` | 참고한 유사 프로젝트 |

## 테스트

```bash
python -m unittest discover -s tests -t .
```
