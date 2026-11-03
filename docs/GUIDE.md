# PromptMaker 사용 가이드

## 1. 설치

- 파이썬 3.9 이상. **필수 외부 패키지 없음** (표준 라이브러리만 사용).
- 저장소 폴더에서 바로 쓰거나 `pip install -e .` 로 설치합니다.
- 구글 번역을 더 안정적으로 쓰려면 선택적으로 `pip install deep-translator`.

```bash
cd F:\AI_Dev\PromptMaker
pip install -e .
```

## 2. 사용 방법 3가지

### 2-1. 파이썬 코드

```python
from promptmaker import PromptMaker, GenerateOptions

pm = PromptMaker()

# 기본: 이미지용, 타입별 랜덤
print(pm.generate("비 오는 날 카페 창가에 앉은 여자").prompt)

# 옵션 지정
opts = GenerateOptions(
    prompt_type="video",                 # image | video
    mode="random_by_element",            # 아래 '모드' 참고
    template="video_cinematic",
    selections={"nationality": "Korean", "location": "on a sunny tropical beach"},
    random_elements=["attire", "lighting", "camera_movement"],  # attire = 의상 요소 전체
    content_rating="sfw",                # sfw | suggestive | explicit
    seed=1234,                           # 같은 시드 = 같은 결과
)
r = pm.generate("노을 지는 해변", opts)
print(r.prompt)        # 최종 영문 프롬프트
print(r.english_input) # 번역된 입력
print(r.lexicon)       # 키워드, 일치 항목, 관련 단어
print(r.picks)         # 요소별로 선택된 값
print(r.warnings)      # 번역 실패, 나이 보정 등 안내

# 여러 개 한 번에
for r in pm.generate_batch("beach", count=5, prompt_type="image"):
    print(r.prompt)
```

간단 호출: `import promptmaker; promptmaker.generate("카페", prompt_type="image")`

### 2-2. 명령줄

```bash
python -m promptmaker "해변 노을" --type image --count 3
python -m promptmaker "cafe" --mode manual --template portrait_default --set nationality=Japanese --set age=24
python -m promptmaker "" --mode random_by_element --template fine_art --random lighting,location
python -m promptmaker --list            # 템플릿/요소 목록
python -m promptmaker "카페" --json      # 상세 결과
```

### 2-3. HTML 테스트 페이지

**Windows: `run_server.bat` 을 더블클릭**하면 서버가 켜지고 브라우저가 자동으로 열립니다. 포트를 바꾸려면 `run_server.bat 8800`. 저장소 폴더에 `user_data/` 가 있으면 자동으로 사용합니다.

```bash
python -m promptmaker.server            # 또는 promptmaker-server
```

브라우저에서 `http://127.0.0.1:8765` 를 엽니다. 옵션, 템플릿, 요소 고정값, 요소별 랜덤 체크, 수위, 시드, 생성 개수를 고를 수 있고, 결과 복사와 상세 정보(번역문, 키워드, 관련 단어, 선택된 요소)를 볼 수 있습니다.
데이터 파일을 수정한 뒤에는 **"데이터 다시 읽기"** 버튼을 누르세요.

## 3. ComfyUI 커스텀 노드

1. 저장소를 `ComfyUI/custom_nodes/PromptMaker` 로 복사하거나 `git clone` 합니다.
2. ComfyUI 를 재시작하면 `PromptMaker` 카테고리에 노드 2개가 생깁니다.

| 노드 | 설명 |
|---|---|
| **PromptMaker 프롬프트 생성** | 입력: user_input, prompt_type, mode, template, content_rating, seed, translate, use_related_words, (선택) selections_json, random_elements, prefix, suffix / 출력: prompt, info_json, seed |
| **PromptMaker 데이터 다시 읽기** | JSON/TXT 수정 후 재시작 없이 다시 읽기 |

- `prompt` 출력을 CLIP Text Encode 의 text 입력에 연결합니다.
- `selections_json` 예: `{"nationality": "Korean", "attire": "a traditional Korean hanbok"}`
- `random_elements` 예: `lighting, location` (모드가 `random_by_element` 일 때)
- seed 의 "control after generate" 를 randomize 로 두면 실행할 때마다 새 프롬프트가 나옵니다.
- 사용자 데이터를 따로 두려면 노드 폴더 안에 `user_data/` 를 만드세요 (아래 5-4 참고).

## 4. 옵션 설명

| 옵션 | 값 | 설명 |
|---|---|---|
| `prompt_type` | `image` / `video` | 프롬프트 종류. 해당 종류의 템플릿만 사용 |
| `mode` | 아래 표 | 랜덤 방식 |
| `template` | 템플릿 id / `random` | 사용할 템플릿 |
| `selections` | `{요소: 값}` | 요소 고정값. **모든 모드에서 최우선** 적용. `"a\|b"` 로 여러 값 지정 가능(`{요소:2}` 슬롯용) |
| `random_elements` | 요소 이름 목록, `*` = 전부 | 요소별 랜덤 모드에서 랜덤으로 뽑을 요소 |
| `content_rating` | `sfw` / `suggestive` / `explicit` | 수위. 해당 수위 이하의 항목/템플릿만 사용 |
| `seed` | 정수 / `None` | 같은 시드 + 같은 옵션 = 같은 결과 |
| `translate` | bool | 영어가 아니면 번역 |
| `use_related_words` | bool | 확률 기반 관련 단어 사용 |
| `model_profile` | `flux` / `sdxl` / `wan_video` … | 결과에 모델 예시를 첨부 (추후 LLM 옵션용) |

### 모드

| 모드 | 한글 이름 | 동작 |
|---|---|---|
| `random_by_type` (기본) | 타입별 랜덤 | 선택한 종류(이미지/영상) 안에서 템플릿과 모든 요소를 랜덤 |
| `random` | 전체 랜덤 | 종류, 템플릿, 요소 모두 랜덤 |
| `random_by_element` | 요소별 랜덤 | 지정한 템플릿 사용, `random_elements` 의 요소만 랜덤, 나머지는 기본값 |
| `manual` | 직접 선택 | 지정한 템플릿 + `selections` 값, 나머지는 기본값 (항상 같은 결과) |

**기본값 우선순위:** `selections` > 사용자 요청에서 뽑은 값 > 템플릿 `defaults` > 요소 파일의 `default` > 목록의 첫 번째 항목

### 사용자 요청이 결과를 결정합니다 (요청 우선 규칙)

요청 문장에서 다음을 먼저 뽑아 **고정**하고, 랜덤은 요청에 없는 부분만 채웁니다.

| 요청에서 찾는 것 | 예 | 결과 |
|---|---|---|
| 나이 | `20살`, `25-year-old`, `20대 후반`, `in her late twenties` | `age` 고정 (20 미만은 20) |
| 국적 | `일본`, `Japanese`, `Japan` | `nationality` 고정 |
| 인물 수/성별 | `여자`, `모델`(1명, 남성 없음), `two women`, `커플`, `남자친구` | `casting` 고정, 맞는 템플릿만 선택 |
| 호칭 | `model`, `idol`, `actress` | 주어가 "woman" 대신 "model" 등으로 |
| 종류 | `사진/화보/photo` → 이미지, `영상/video` → 영상 | 전체 랜덤 모드에서 종류 제한 |
| 장소, 시간/날씨, 스타일, 의상, 포즈/동작 등 | `해변`, `비 오는 날`, `그라비아`, `한복`, `walking` | 그 단어가 들어간 항목 중에서만 선택 |
| 템플릿 | 템플릿 JSON 의 `"keywords"` 와 겹치는 단어 (`화보/editorial/gravure` → 패션 화보) | 점수가 높은 템플릿 우선 |

요청 문장을 그대로 끝에 붙이지 않고, 요소로 쓰이지 않은 단어만 "The scene also features ..." 로 덧붙입니다.
템플릿 JSON 에 `"keywords": ["editorial", "gravure", ...]` 를 추가하면 템플릿 선택에 반영됩니다.

## 5. 데이터 파일 편집 (템플릿 / 요소)

모든 데이터는 `promptmaker/data/` 아래에 있으며 메모장으로 편집할 수 있습니다 (UTF-8 로 저장).

### 5-1. 요소 파일 (TXT)

`elements/<요소이름>.txt` — 한 줄에 항목 하나.

```text
# '#' 으로 시작하면 주석
soft golden-hour sunlight
3::diffused natural window light      ← 가중치 3 (3배 자주 선택)
giving {her} skin a warm, glowing tone ← 다른 슬롯 사용 가능
{red|crimson|scarlet} satin dress      ← 인라인 랜덤 변형
```

- 파일 이름의 두 번째 부분은 **수위**입니다: `attire_state.txt`(sfw), `attire_state.suggestive.txt`, `attire_state.explicit.txt`.
  `content_rating` 이 그 수위 이상일 때만 해당 파일의 항목이 사용됩니다.
- `*.explicit.txt` 파일은 비워 두었습니다. 필요하면 사용자가 직접 작성하세요.
- 새 파일을 만들면 그 이름이 곧 새 요소가 되며, 템플릿에서 `{파일이름}` 으로 사용합니다.
- 미성년자를 연상시키는 항목은 불러올 때 자동으로 제외됩니다.

**항목 작성 규칙 (자연스러운 문장을 위해)**

| 요소 | 문법 형태 | 템플릿 속 예 |
|---|---|---|
| `photo_style`, `video_style` | 명사구 (관사 없이) | `A {photo_style} of ...` |
| `nationality`, `beauty`, `male_look`, `mood` | 형용사 | 주어 구문에 자동 결합 |
| `age` | 숫자 (20 이상) | `20-year-old` 로 자동 변환 |
| `hair`, `makeup`, `expression`, `lighting` | 명사구 | `{She} {has} {hair}` |
| `top`, `bottom`, `dress`, `swimwear`, `sleepwear`, `outerwear`, `footwear`, `accessory` 등 의상 | 명사구 (관사 포함) | `{attire}` 슬롯이 자동 조합 |
| `skin` | 형용사구 | `... and {skin} skin` |
| `pose`, `action`, `motion`, `interaction` | 현재분사구 (-ing) | `{She} {is} {pose}` |
| `location`, `time_weather` | 전치사구 (in/on/at ...) | `The scene takes place {location}` |
| `camera_movement` | 3인칭 동사구 | `The camera {camera_movement}` |
| `ambient_motion` | `while ...` 절 | `..., {ambient_motion}` |

항목 안에서 she/her 대신 `{she}` `{her}` `{her_obj}` 를 쓰면 인물이 여러 명일 때 they/their/them 으로 자동 변환됩니다.

### 5-1-1. 의상 요소와 `{attire}` 자동 조합

예전의 `attire.txt` 한 파일은 아래처럼 나뉘었습니다. 템플릿의 `{attire}` 슬롯은 이제 이 요소들을 **문법에 맞게 조합**합니다.

| 단계 | 사용 요소 | 예 |
|---|---|---|
| 1. 의상 구성 고르기 | `outfit_type` (top_bottom, dress, swimwear, sleepwear, traditional, sportswear, lingerie) | `swimwear` |
| 2. 본 의상 | top_bottom 이면 `top` + `bottom`, 그 외에는 같은 이름의 요소 | `a white string bikini` |
| 3. 겉옷 (선택) | `outerwear` — 추울 때(winter/rain/snow) 60%, 평소 15%, 여름·수영복·잠옷에는 없음 | `... under a long camel wool coat` |
| 4. 신발·액세서리 (선택) | `footwear` 약 50%, `accessory` 약 40% | `... with strappy sandals and a sun hat` |

- 요청에 "비키니"처럼 특정 의상이 나오면 그 구성이 고정됩니다.
- `selections` 에 `attire` 를 직접 넣으면 조합 대신 그 문장을 그대로 씁니다.
- 요소별 랜덤 모드에서 `attire` 를 랜덤으로 지정하면 모든 의상 요소가 랜덤이 됩니다.

### 5-1-2. 관계 규칙: 어울리는 항목끼리만 나오게 하기 (태그)

요소는 **장면 순서**대로 선택됩니다: 장소 → 시간/날씨 → 사진 스타일 → 의상 → 의상 상태 → 포즈 → 동작 → 표정 → 조명 → 카메라 → 분위기.
선택된 항목의 태그가 "장면 정보"에 쌓이고, 다음 요소는 그 장면에 맞는지 검사를 받습니다.

TXT 항목 줄 끝에 태그를 직접 붙일 수 있습니다 (붙인 태그는 문장에 나오지 않습니다).

```text
lying on a beach towel #summer +beach !night
```

| 표기 | 뜻 |
|---|---|
| `#태그` | 이 항목이 장면에 이 태그를 **추가** |
| `+태그` | 장면에 이 태그가 **있어야** 선택 가능 (여러 개면 모두 필요) |
| `!태그` | 장면에 이 태그가 있으면 **선택 불가** |

태그를 안 붙여도 **자동 태깅**이 됩니다 (`lexicon/tag_keywords.json`).

- `tags`: 문장에 단어가 있으면 태그를 붙입니다. 예: "bikini" → `swim`, `summer`.
- `props` (소품): 장소/시간과 의상 요소에서는 "제공", 포즈·동작 등에서는 "필요"로 처리됩니다.
  예: 포즈 "sitting on the edge of the bed" 는 `bed` 가 필요 → 장소가 해변이면 선택되지 않습니다.
- `enforced_props` (hem, sleeves, collar, strap): 장면을 몰라도 항상 검사합니다. 예: "with the sleeves rolled up" 은 소매 있는 옷일 때만.
- 소품 검사(`+`)는 장소가 정해졌을 때(요청에 장소가 있거나 장소가 선택된 뒤)만 적용됩니다.

확률은 `lexicon/relations.json` 으로 정합니다.

- `implies`: 태그가 다른 태그를 포함. 예: `beach` → outdoor, summer, sand, water, sea.
- `conflicts`: 함께 나올 수 없는 짝 → 확률 0. 예: `["beach", "bed"]`, `["swim", "winter"]`, `["day", "night"]`.
- `affinity`: 장면 태그가 있을 때 항목 태그의 가중치 배수. 예: `"beach": {"swim": 8, "formal": 0.2}` → 해변에서는 수영복이 8배, 정장은 0.2배.
- 최종 확률 = 항목 가중치 × (장면 태그별 affinity 의 곱) × 요청 단어 가산점. 규칙을 만족하는 항목이 하나도 없으면 규칙 없이 고르고 경고를 남깁니다.

**점검 명령**: 줄을 추가한 뒤 `python -m promptmaker --check` 를 실행하면 규칙에 없는 태그나, 어떤 장소/의상도 제공하지 않아 선택될 수 없는 `+태그` 를 알려 줍니다.

**선택 이유 보기**: 결과의 `lexicon.trace` 에 요소별 선택 항목과 적용된 배수가 나옵니다 (HTML 페이지의 "선택 과정" 칸에도 표시). 예:
`swimwear: a white string bikini  [beach→swim×8, gravure→swim×5]`

### 5-2. 요소 파일 (JSON)

구조화된 데이터가 필요하면 JSON 을 씁니다. 예: `elements/casting.json`

```json
{
  "label": "인물 수 / 성별 구성",
  "default": "one woman",
  "items": [
    {"text": "one woman", "female": 1, "male": 0, "weight": 6},
    {"text": "a woman and a man", "female": 1, "male": 1, "weight": 3},
    "문자열만 써도 됩니다"
  ]
}
```

`"replace": true` 를 넣으면 앞서 읽은 같은 요소의 항목을 지우고 대체합니다 (사용자 데이터 폴더에서 유용).

### 5-3. 템플릿 (JSON)

`templates/<이름>.json`

```json
{
  "id": "portrait_default",
  "name": "기본 인물 사진",
  "type": "image",
  "rating": "sfw",
  "description": "설명",
  "casting_filter": {"male_min": 0, "male_max": 1, "female_max": 2},
  "defaults": {"photo_style": "cinematic film still"},
  "sentences": [
    {"text": "A {photo_style} of {subject}."},
    {"text": "{She} {has} {hair}, {makeup}, and {skin} skin."},
    {"text": "{She} {is} {interaction}.", "when": "male > 0"},
    {"text": "{camera}, {color_grade}.", "probability": 0.7},
    {"options": ["문장 A {location}.", "문장 B {location}."]},
    {"text": "{lens?} {color_grade:2}."}
  ]
}
```

| 키 | 설명 |
|---|---|
| `sentences[].text` | 문장. `{요소}` 슬롯을 채웁니다 |
| `sentences[].options` | 여러 문장 중 하나를 랜덤 선택 |
| `sentences[].when` | 조건. 사용 가능 변수: `female`, `male`, `video`, `image`, `rating`(0~2), `has_input` |
| `sentences[].probability` | 0~1. 이 확률로만 문장 포함 |
| `sentences[].rating` | 이 수위 이상일 때만 문장 포함 |
| `casting_filter` | 인물 구성 제한 (`female_min/max`, `male_min/max`) |
| `defaults` | 이 템플릿에서의 요소 기본값 |
| `rating` | 템플릿 자체의 최소 수위 (낮은 수위에서는 랜덤 선택 대상에서 제외) |

**슬롯 문법**

| 슬롯 | 의미 |
|---|---|
| `{location}` | 요소에서 1개 선택 |
| `{attire:2}` | 서로 다른 2개 선택 후 "A and B" 로 연결 |
| `{lens?}` | 선택 사항. 비어 있어도 문장을 버리지 않음 (기본은 슬롯이 비면 문장 생략) |
| `{a\|b\|c}` | 인라인 랜덤 |

**자동 계산 슬롯**

| 슬롯 | 1명 | 여러 명 |
|---|---|---|
| `{subject}` | a stunning 24-year-old Korean woman (and a handsome adult man) | two stunning 24-year-old Korean women |
| `{female_subject}`, `{male_subject}` | 여성/남성 부분만 | |
| `{she}` `{She}` `{her}` `{Her}` `{her_obj}` `{herself}` | she, She, her, Her, her, herself | they, They, their, Their, them, themselves |
| `{is}` `{has}` `{wears}` `{looks}` `{s}` | is, has, wears, looks, s | are, have, wear, look, (빈칸) |
| `{partner}` `{Partner}` `{him}` `{his}` `{is_m}` | the man, The man, him, his, is | the men, The men, them, their, are |
| `{age}` | 나이 숫자 | |
| `{user_details}` | 사용자 입력 + 관련 단어 문장. 템플릿에 없으면 맨 끝에 자동 추가 | |

### 5-4. 사용자 데이터 폴더 (업데이트해도 내 수정이 유지되게)

기본 데이터를 직접 고쳐도 되지만, 별도 폴더를 두면 저장소를 업데이트해도 내 데이터가 덮어쓰이지 않습니다.

```
my_data/
  elements/location.txt        ← 기본 location 항목에 '추가'
  elements/attire.json         ← "replace": true 면 '대체'
  templates/my_template.json   ← 새 템플릿 (같은 id 면 기본 템플릿 대체)
  lexicon/related_words.json   ← 키워드 추가/대체
```

```python
pm = PromptMaker(user_data_dir="my_data")
```

또는 환경변수 `PROMPTMAKER_USER_DATA=my_data`, CLI/서버는 `--user-data my_data`, ComfyUI 는 노드 폴더 안의 `user_data/`.

### 5-5. 관련 단어 사전

`lexicon/related_words.json` — 입력 키워드에 대해 확률적으로 관련 단어를 뽑습니다.

```json
"rain": {
  "aliases": ["rainy", "drizzle"],
  "adjectives": [["moody", 0.8], ["wet", 0.7]],
  "verbs": [["raindrops trickling down the glass", 0.6]],
  "nouns": [["rain-streaked windows", 0.5]],
  "elements": {"location": ["in a rainy city street with glowing reflections"]}
}
```

- `[단어, 확률]`: 각 단어가 그 확률로 선택됩니다 (형용사/동사/명사 각각 최대 3개).
- `elements`: 해당 요소에서 이 항목을 우선 선택합니다 (가중치 40배, 목록에 없으면 후보로 추가).
- 또한 입력 키워드가 들어 있는 요소 항목은 자동으로 가중치 60배가 됩니다.

`lexicon/ko_en.json` 은 구글 번역에 실패했을 때 쓰는 간단한 오프라인 한영 단어 사전입니다.

## 6. 번역

- 기본: `GoogleTranslator` — `deep-translator` 가 설치되어 있으면 사용, 없으면 구글 번역 공개 엔드포인트를 직접 호출. 실패하면 오프라인 사전으로 대체하고 `warnings` 에 기록합니다.
- 교체 가능: `translate(text) -> str` 메서드를 가진 객체면 무엇이든 됩니다.

```python
from promptmaker import PromptMaker, NoTranslator
pm = PromptMaker(translator=NoTranslator())   # 번역 끄기

class MyTranslator:
    def translate(self, text): return my_api(text)
pm = PromptMaker(translator=MyTranslator())
```

## 7. 고정 입력 (시스템 프롬프트, 모델 예시) 과 추후 LLM 옵션

- `data/system/system_prompt.txt`, `data/model_examples/*.txt` 는 결과의 `system_prompt`, `model_example` 로 함께 전달됩니다.
- 현재 생성 과정은 LLM 을 쓰지 않습니다. 추후 최소한의 LLM 보정을 붙이려면:

```python
class MyLLM:
    def refine(self, prompt, system_prompt, examples):
        return call_llm(system_prompt, examples, prompt)

pm = PromptMaker(llm=MyLLM())
pm.generate("카페", use_llm=True)
```

### 7-1. 이미지 분석으로 요소 자동 추가 (선택 기능)

이미지를 비전 모델로 분석해 요소별 영문 문구를 JSON 으로 받고, 기존 요소 txt 와 **중복되지 않는 항목만** 파일 끝에 추가합니다. 프롬프트 생성 자체는 여전히 LLM 을 쓰지 않으며, 이 기능을 쓸 때만 모델이 필요합니다.

처리 과정: 이미지 → 비전 모델 (기본 `Qwen/Qwen3-VL-4B-Instruct`) → `{"hair": ["..."], "location": ["..."]}` → 파싱 → 안전 검사 + 중복 확인 → `elements/<요소>.txt` 끝에 추가

**필요 패키지**: `pip install torch transformers pillow` (transformers 4.57 이상). ComfyUI 파이썬에는 보통 이미 설치되어 있습니다.

**모델 설정** (`config/vision.json`, HTML 페이지의 "모델 설정" 에서 저장 가능, 깃에는 올라가지 않음)

| 항목 | 설명 |
|---|---|
| `backend` | `transformers` (로컬 모델 폴더, 기본) 또는 `openai` (Ollama, LM Studio, llama.cpp `llama-server` 등 OpenAI 호환 서버. GGUF 모델은 이 방식으로 연결) |
| `model_path` | 모델 폴더 위치 (`config.json` 과 `.safetensors` 가 있는 폴더). 비우면 `model_id` 를 자동 다운로드 |
| `model_id` | Hugging Face 모델 id, 또는 `openai` 백엔드에서 서버의 모델명. 다른 모델(예: `Qwen/Qwen3-VL-8B-Instruct`)로 바꿀 수 있음 |
| `api_base`, `api_key` | `openai` 백엔드의 서버 주소 / 키 |
| `device`, `dtype` | `auto` / `cuda` / `cpu`, `auto` / `bfloat16` / `float16` / `float32` |
| `max_items_per_key` | 요소당 최대 문구 수 (기본 3) |
| `skip_keys` | 분석하지 않을 요소 (기본: casting, age, outfit_type, nationality, camera_movement, pacing) |

**중복 확인**: 해당 요소의 모든 수위 파일(`<요소>.txt`, `.suggestive.txt`, `.explicit.txt`)과 사용자 데이터 폴더의 줄을 읽어, 가중치(`3::`), 줄 끝 태그(`#tag +tag !tag`), `{ }`, 앞의 a/an/the, 대소문자, 공백, 문장부호를 무시하고 비교합니다. 미성년자 연상 표현은 추가하지 않고, txt 파일이 없는 요소(JSON 전용 등)도 건너뜁니다. 기존 줄은 수정하지 않고 `# 이미지 분석 자동 추가 (날짜)` 주석 아래에 덧붙이기만 합니다.

**사용 방법**

- HTML 페이지: 아래쪽 "이미지 분석 → 요소 자동 추가" 에서 "찾아보기" 로 모델 폴더를 고르고, 이미지를 끌어다 놓고(또는 Ctrl+V) "이미지 분석". 새 항목만 체크되어 나오며 문구를 고친 뒤 "선택 항목 txt 에 추가".
- ComfyUI: `PromptMaker 이미지 분석 → 요소 추가` 노드. `IMAGE` 입력, `model_path` 에 모델 폴더, `write_to_txt` 를 끄면 미리보기만 합니다. `save_as_default` 를 켜면 설정을 `config/vision.json` 에 저장합니다.
- 명령줄:

```bash
python -m promptmaker.vision photo.jpg                       # 미리보기 (추가 안 함)
python -m promptmaker.vision photo.jpg --apply               # 새 항목을 txt 에 추가
python -m promptmaker.vision photo.jpg --model-path "F:\models\Qwen3-VL-4B-Instruct" --save-config
python -m promptmaker.vision photo.jpg --backend openai --api-base http://127.0.0.1:11434/v1 --model-id qwen3-vl:4b
```

- 파이썬:

```python
from promptmaker import PromptMaker
from promptmaker.vision import ImageAnalyzer, load_config

pm = PromptMaker()
an = ImageAnalyzer(pm, load_config())
result = an.analyze("photo.jpg")      # {"elements": {요소: [{"text", "status"}]}, "new": {...}, "raw": 모델 응답}
report = an.apply(result["new"])       # {"added": {...}, "skipped": {...}, "added_count": n}
```

분석 지시문은 `promptmaker/data/vision/analyze_prompt.txt` 에서 고칠 수 있습니다. 요소 목록과 예시 문구는 현재 요소 파일에서 자동으로 채워지므로 요소를 새로 추가해도 따로 고칠 필요가 없습니다.

## 8. 안전 규칙 (성인 전용)

- 모든 인물은 20세 이상: `age` 값이 20 미만이면 20으로 보정하고 경고를 남깁니다. 주어 문장에는 `adult man` 등 성인임이 명시됩니다.
- 입력/고정값에 미성년자 연상 표현(teen, schoolgirl, 여고생, 17살 등)이 있으면 `SafetyError` 로 거부합니다.
- 데이터 파일에 그런 항목이 있으면 불러올 때 제외됩니다.
- 규칙은 `promptmaker/safety.py` 에 있습니다.

## 9. GitHub 자동 관리 (`tools/github_auto.py`)

토큰은 **실행 시점에 파일에서만 읽고**, 화면/로그/명령줄에 남기지 않습니다 (git 은 `GIT_ASKPASS` 로 전달, 오류 메시지에서도 가림 처리).

토큰 파일 경로 우선순위: `--token-file` → 환경변수 `PROMPTMAKER_TOKEN_FILE` → `<저장소>/token/git_token.txt` → `F:\AI_Dev\PromptMaker\token\git_token.txt`
`token/` 폴더는 `.gitignore` 에 등록되어 커밋되지 않습니다.

```bash
python tools/github_auto.py status                     # 로컬 상태 + 연결 계정 확인
python tools/github_auto.py pull
python tools/github_auto.py push
python tools/github_auto.py auto -m "요소 추가" --branch feature/elements --pr   # add/commit/push/PR 한 번에
python tools/github_auto.py pr --title "제목" --body "설명" --base main
python tools/github_auto.py prs                        # 열린 PR 목록
python tools/github_auto.py issue --title "버그" --body "내용"
python tools/github_auto.py issues
```

`auto` 는 main 에서 실행하면 직접 푸시하지 않고 `--branch` 로 새 브랜치를 만들도록 요구합니다.

## 10. 문제 해결

| 증상 | 해결 |
|---|---|
| 경고 "구글 번역 실패" | 네트워크/요청 제한. `pip install deep-translator` 또는 영어로 입력. 오프라인 사전으로 키워드는 계속 동작합니다 |
| 수정한 데이터가 반영 안 됨 | `pm.reload()`, HTML "데이터 다시 읽기", ComfyUI "데이터 다시 읽기" 노드 |
| 결과에 `{...}` 가 남음 | 슬롯 이름 오타. 해당 이름의 요소 파일이 있는지 확인 |
| 특정 문장이 빠짐 | 슬롯 요소가 비어 있거나(수위 필터 포함) `when` 조건이 거짓 |
| 문법이 어색함 | 항목의 문법 형태(5-1 표)를 맞추고, she/her 대신 `{she}` `{her}` 사용 |
