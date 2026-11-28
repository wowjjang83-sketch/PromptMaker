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

브라우저에서 `http://127.0.0.1:8765` 를 엽니다. 옵션, 템플릿, 요소 고정값, 요소별 랜덤 체크, 시드, 생성 개수를 고를 수 있고, 결과 복사와 상세 정보(번역문, 키워드, 관련 단어, 선택된 요소)를 볼 수 있습니다.
데이터 파일을 수정한 뒤에는 **"데이터 다시 읽기"** 버튼을 누르세요.

## 3. ComfyUI 커스텀 노드

1. 저장소를 `ComfyUI/custom_nodes/PromptMaker` 로 복사하거나 `git clone` 합니다.
2. ComfyUI 를 재시작하면 `PromptMaker` 카테고리에 노드 2개가 생깁니다.

| 노드 | 설명 |
|---|---|
| **PromptMaker 프롬프트 생성** | 입력: user_input, prompt_type, mode, template, seed, translate, use_related_words, (선택) selections_json, random_elements, prefix, suffix / 출력: prompt, info_json, seed |
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
| `seed` | 정수 / `None` | 같은 시드 + 같은 옵션 = 같은 결과 |
| `translate` | bool | 영어가 아니면 번역 |
| `use_related_words` | bool | 확률 기반 관련 단어 사용 |
| `model_profile` | `krea2` / `qwen2.1` / `moss` / `minimax_h3` / `none` | 대상 모델 형식으로 출력. 비우면 이미지는 `krea2`, 영상은 `minimax_h3` (아래 "대상 모델" 참고) |
| `framing` | `auto` / `extreme_closeup` / `closeup` / `medium` / `cowboy` / `full_body` / `none` | 샷 구도. `auto` 는 요청에서 찾음('얼빡샷' 등), 고르면 요청보다 우선. 아래 7-5 참고 (0.9.0) |
| `body_part` | `""` / `eyes` / `lips` / `hair` / `hands` / `neck` / `shoulders` / `back` / `chest` / `waist` / `hips` / `thighs` / `legs` / `feet` | 신체 부위 클로즈업. 비우면 요청에서 찾음('다리 클로즈업'). 아래 7-6 참고 (0.11.0) |

### 모드

| 모드 | 한글 이름 | 동작 |
|---|---|---|
| `random_by_type` (기본) | 타입별 랜덤 | 선택한 종류(이미지/영상) 안에서 템플릿과 모든 요소를 랜덤 |
| `random` | 전체 랜덤 | 요소 모두 랜덤. 종류는 요청에 "영상/video" 같은 말이 있으면 그 종류, 없으면 선택한 종류 |
| `random_by_element` | 요소별 랜덤 | 지정한 템플릿 사용, `random_elements` 의 요소만 랜덤, 나머지는 기본값 |
| `manual` | 직접 선택 | 지정한 템플릿 + `selections` 값, 나머지는 기본값 (항상 같은 결과) |

템플릿을 직접 고르면(랜덤이 아니면) **모든 모드에서** 그 템플릿을 씁니다.

**기본값 우선순위:** `selections` > 사용자 요청에서 뽑은 값 > 템플릿 `defaults` > 요소 파일의 `default` > 목록의 첫 번째 항목

### 사용자 요청이 결과를 결정합니다 (요청 우선 규칙)

- "하얀 미니스커트", "밝은 피부"처럼 **꾸밈말 + 의상/피부** 로 쓴 말은 그 의상/피부에만 적용됩니다
  ("하얀"이 배경의 흰 스튜디오로 새지 않음). 목록에 같은 항목이 없으면 요청한 말 그대로("a white mini skirt") 씁니다.
- 인물 수를 말하지 않고 한 사람의 외모(의상, 피부)를 적으면 여성 1명으로 생성합니다.

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

- **수위 구분은 없습니다 (0.4.0).** 요소마다 `<요소>.txt` 하나에 모든 항목을 적습니다.
  예전 수위 파일(`attire_state.suggestive.txt`, `attire_state.explicit.txt` 등)이 사용자 데이터 폴더에 남아 있으면
  `PromptMaker(user_data_dir=...)` 를 만들 때 자동으로 `<요소>.txt` 끝에 합쳐지고(중복 제외, `# 수위 통합: ... 에서 옮김` 주석) 원래 파일은 지워집니다.
  직접 합치려면 `from promptmaker.data_loader import merge_legacy_rating_files; merge_legacy_rating_files("폴더")`.
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

**수정한 TXT 를 적용하려면:** 테스트 페이지의 "데이터 다시 읽기" 버튼(또는 `/api/reload`)을 누르면 됩니다. 서버 재시작은 필요 없습니다.
서버 재시작은 파이썬 코드를 업데이트했을 때(`git pull` 후)만 필요합니다.
포즈·동작 등에 완성 문장("Standing at the edge of a rooftop, she gazes at ...")을 넣으면 앞부분("standing at the edge of a rooftop")만 씁니다.
그 안에 장소(rooftop, field, beach ...)가 있으면 그 장소일 때만 선택됩니다. 의상 상태에 옷 이름(blouse, skirt ...)이 있으면 그 옷을 입었을 때만 선택됩니다.
불러올 때 자동으로 빠지는 줄: 미성년자 관련 단어, 20 미만 나이, 쉼표로 단어만 나열한 태그 줄(예: "Short, Firm, Square Face, Fair Skin, ...").

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
| `sentences[].when` | 조건. 사용 가능 변수: `female`, `male`, `video`, `image`, `has_input` |
| `sentences[].probability` | 0~1. 이 확률로만 문장 포함 |
| `casting_filter` | 인물 구성 제한 (`female_min/max`, `male_min/max`) |
| `defaults` | 이 템플릿에서의 요소 기본값 |

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

## 7. 대상 모델 형식 (모델별 프롬프트 예시)

생성된 프롬프트를 사용할 모델의 권장 형식으로 바꿔서 출력합니다. 라이브러리, 테스트 페이지("대상 모델"), ComfyUI 노드(`model_profile` 입력, `negative_prompt` 출력), 명령줄(`--model`) 모두 같습니다.

| id | 모델 | 종류 | 형식 |
|---|---|---|---|
| `krea2` | Krea 2 | 이미지 (**기본**) | 자연스러운 문단 1~2개 |
| `qwen2.1` | Qwen-Image 2.1 | 이미지 | 완전한 문장. 네거티브 칸이 없어 "No text, no watermark, no logos." 를 끝에 붙임 |
| `moss` | MOSS-Image | 이미지 | 품질 → 인물 → 일반 태그 순서의 쉼표 나열 + 별도 네거티브 프롬프트 |
| `minimax_h3` | MiniMax H3 (Hailuo 3) | 영상 (**기본**) | `integrated_multimodal_description:` / `overall_soundscape:` / `non_diegetic_music:` 세 부분 |

- 비워 두면 이미지는 `krea2`, 영상은 `minimax_h3` 형식이 됩니다. `none` 이면 변환하지 않습니다.
- 결과의 `prompt` 는 변환된 프롬프트, `base_prompt` 는 변환 전 문장, `negative_prompt` 는 네거티브 프롬프트(MOSS 등)입니다.
- 형식은 `data/model_profiles/<id>.json` 에서 고칩니다. 주요 항목: `format`(prose / tags / h3), `default_for`(image / video), `max_chars`, `prefix`, `suffix`, `negative`, `inline_exclusions`, `tag_order`, `quality_tags`, `replace`, `music`, `sounds`.
- 새 모델은 JSON 파일을 하나 추가하면 됩니다. 예시 프롬프트는 `data/model_examples/<id>.txt` 에 두면 결과의 `model_example` 로 함께 전달됩니다.

```python
r = pm.generate("해변 비키니 화보", model_profile="moss")
print(r.prompt)           # masterpiece, best quality, ...
print(r.negative_prompt)  # worst quality, bad anatomy, ...
```

## 7-0. 고정 입력 (시스템 프롬프트, 모델 예시) 과 추후 LLM 옵션

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

**GPU 로 쓰려면**: HTML 테스트 서버를 CUDA 용 torch 가 있는 파이썬으로 실행해야 합니다. `config/python_path.txt` 첫 줄에 그 파이썬 경로(예: `F:\AI_Dev\ComfyUI_20261014\python_embeded\python.exe`)를 적거나 환경변수 `PROMPTMAKER_PYTHON` 을 지정한 뒤 `run_server.bat` 을 실행하세요. 모델 설정 화면에 현재 서버 파이썬의 GPU 사용 가능 여부가 표시됩니다. 이미 받아 둔 Hugging Face 모델은 인터넷 접속 없이 캐시에서 읽습니다.

**모델 설정** (`config/vision.json`, HTML 페이지의 "모델 설정" 에서 저장 가능, 깃에는 올라가지 않음)

| 항목 | 설명 |
|---|---|
| `backend` | `transformers` (로컬 모델 폴더, 기본) 또는 `openai` (Ollama, LM Studio, llama.cpp `llama-server` 등 OpenAI 호환 서버. GGUF 모델은 이 방식으로 연결) |
| `model_path` | 모델 폴더 (`config.json` 과 `.safetensors` 가 있는 폴더) 또는 **단일 `.safetensors` 파일** (ComfyUI 용 FP8 파일 포함, 자동으로 풀어서 읽음). 비우면 `model_id` 를 자동 다운로드 |
| `model_id` | Hugging Face 모델 id 또는 폴더. 다른 모델(예: `Qwen/Qwen3-VL-8B-Instruct`)로 바꿀 수 있음. 단일 파일을 쓸 때는 설정·토크나이저를 여기서 가져오므로 같은 구조의 원본 모델로 두세요. `openai` 백엔드에서는 서버의 모델명 |
| `api_base`, `api_key` | `openai` 백엔드의 서버 주소 / 키 |
| `device`, `dtype` | `auto` / `cuda` / `cpu`, `auto` / `bfloat16` / `float16` / `float32` |
| `max_items_per_key` | 요소당 최대 문구 수 (기본 3) |
| `skip_keys` | 분석하지 않을 요소 (기본: casting, age, outfit_type, nationality, camera_movement, pacing) |
| `unload_after` | 분석이 끝나면 모델을 내리고 GPU 캐시를 비움 (기본 켬). 한 장 분석은 끝날 때마다, 대기열은 모두 끝났을 때 한 번 내립니다. 끄면 다음 분석이 빨라지는 대신 메모리를 계속 차지합니다 |
| `quantization` | `auto`(기본): GPU 여유 메모리가 10.5GB 이상이면 bf16(약 9GB, 생성이 더 빠름), 아니면 FP8(약 5GB). `fp8` / `none`(bf16) 로 고정 가능. FP8 은 RTX 40 시리즈 이상에서 하드웨어 FP8 연산을 씁니다 |
| `prefilter` | 먼저 사람·남자·커플이 있는지 짧게 묻고, 없는 쪽의 요소(머리·의상·포즈 등)는 아예 묻지 않음 (기본 켬). 사람이 없는 사진에서 답이 1/3 로 줄어 빨라집니다. 묶음은 `data/vision/key_groups.json` |
| `outfit_pass` | (0.15.0) 사람이 있으면 분석 뒤에 **의상만** 한 번 더 짧게 묻습니다 (기본 켬, 몇 초 추가). 요소 37개를 한꺼번에 물으면 작은 모델이 겹쳐 입은 옷(상의 위 코르셋, 벨트, 스타킹 등)을 빠뜨려서, 머리부터 발끝까지 옷만 다시 물어 빠진 의상을 채웁니다. 지시문은 `data/vision/outfit_prompt.txt` |

**의상 분류 (0.15.0)**: 요소 설명은 한글이라 의상 요소 옆에 영어 안내(`vision.KEY_HINTS`, 예: lingerie = bra, panties, corset ...)를 붙여 묻습니다. `attire_state`(의상 상태)에 의상 이름만 있는 문구('tight red ribbed long-sleeve top')가 오면 요소 검사가 해당 의상 요소(top)로 옮깁니다. 'white blouse partially unbuttoned' 처럼 상태가 붙은 문구는 그대로 둡니다. 분석 결과(`analyze()`)의 `skipped` 에는 저장하지 않은 항목(이미 있음 `duplicate`, 문장 조각 등)이 이유와 함께 들어갑니다.

**중복 확인**: 해당 요소의 `<요소>.txt`(기본 데이터 + 사용자 데이터 폴더)의 줄을 읽어, 가중치(`3::`), 줄 끝 태그(`#tag +tag !tag`), `{ }`, 앞의 a/an/the, 대소문자, 공백, 문장부호를 무시하고 비교합니다. 미성년자 연상 표현은 추가하지 않고, txt 파일이 없는 요소(JSON 전용 등)도 건너뜁니다. 기존 줄은 수정하지 않고 `# 이미지 분석 자동 추가 (날짜)` 주석 아래에 덧붙이기만 합니다.

**사용 방법**

- HTML 페이지: 아래쪽 "이미지 분석 → 요소 자동 추가" 에서 "찾아보기" 로 모델 폴더나 파일을 고르고, 이미지를 **여러 장** 끌어다 놓으면(또는 클릭해서 고르기, Ctrl+V) 대기열에 들어가 차례로 분석합니다. 분석 중에도 더 넣을 수 있고, "대기열 중지" 로 멈춥니다. 대기열에서 이미지를 누르면 그 결과가 나오며, 새 항목만 체크되어 있으니 문구를 고친 뒤 "선택 항목 txt 에 추가". "새 요소 항목을 검토 없이 바로 txt 에 추가" 를 켜면 대기열이 알아서 추가합니다.
- ComfyUI: `PromptMaker 이미지 분석 → 요소 추가` 노드. `image` 입력(배치면 여러 장)이나 `image_folder`(폴더의 이미지 전부)를 대기열로 차례로 분석, `full_prompt`/`save_full_prompt` 로 풀 프롬프트 생성·저장(출력 `full_prompt`), `model_path` 에 모델 폴더나 `.safetensors` 파일, `write_to_txt` 를 끄면 미리보기만 합니다. `save_as_default` 를 켜면 설정을 `config/vision.json` 에 저장합니다.
- 명령줄:

```bash
python -m promptmaker.vision photo.jpg                       # 미리보기 (추가 안 함)
python -m promptmaker.vision photo.jpg --apply               # 새 항목을 txt 에 추가
python -m promptmaker.vision a.jpg b.jpg F:\photos --apply   # 여러 장/폴더를 차례로 분석
python -m promptmaker.vision photo.jpg --full-prompt         # 풀 프롬프트도 생성해서 보여 줌
python -m promptmaker.vision F:\photos --save-full-prompt    # 풀 프롬프트를 파일에 저장 (경로를 주면 그 파일)
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

# 여러 장: 대기열 (분석 중에 다른 스레드에서 add 해도 이어서 처리)
from promptmaker.vision import AnalysisQueue
q = AnalysisQueue(an, auto_apply=True, auto_save_full_prompt=True)
q.add_many(["a.jpg", "b.jpg"]); q.add_folder("F:/photos")
for item in q.run(progress=lambda item, p: print(item.name, p)):
    print(item.status, item.result and item.result["full_prompt"])

# 풀 프롬프트: 설정의 full_prompt=True 면 result["full_prompt"] 에 한 문단이 들어옴
an.save_full_prompts([result["full_prompt"]])   # 중복·미성년 표현 제외하고 output/full_prompts.txt 에 한 줄씩
```

**풀 프롬프트**: 설정에서 "풀 프롬프트도 생성" (`full_prompt`) 을 켜면 같은 분석 한 번에 요소 목록과 함께 이미지를 묘사하는 영문 프롬프트 한 문단을 받습니다. 저장 버튼(또는 자동 저장)을 누르면 `output/full_prompts.txt` (설정 `full_prompt_file` 로 변경) 에 한 줄씩, 이미 있는 프롬프트와 미성년자 연상 표현은 빼고 덧붙입니다. 출력이 길어지므로 분석 시간이 조금 늘어납니다.

**속도** (RTX 4070 Ti, 4B 모델, 큰 사진 기준): 모델이 올라와 있으면 한 장 6~20초 (사람 없는 사진 약 6초, 인물 사진 9~20초), 분석마다 모델을 다시 불러오면 3~5초 (디스크에서 처음 읽을 때는 10초 이상) 더 걸립니다. 예전에 몇 분씩 걸렸던 것은 큰 사진에서 GPU 12GB 가 넘쳐 시스템 RAM 을 쓰던 탓으로, 이제 여유가 없으면 FP8 로 올립니다.

**모델 언로드**: 기본으로 분석(한 장)이나 대기열이 끝나면 모델을 내리고 GPU 캐시를 비웁니다. HTML 페이지의 "분석이 끝나면 모델 언로드" 체크, ComfyUI 노드의 `unload_when_done`, 파이썬의 `analyze(..., unload=False)` / `AnalysisQueue(..., unload_when_done=False)` 로 끌 수 있고, `vision.unload_backends()` 로 언제든 직접 내릴 수 있습니다. 다시 분석하면 모델을 새로 불러오므로 첫 장이 느려집니다.

**진행 상황**: 분석 중에 단계(모델 로딩 → 준비 → 생성 → 해석·중복 확인), 입력/출력 토큰 수, 생성 속도(tok/s), 경과 시간, 남은 예상 시간을 보여 줍니다. 예상 시간은 첫 분석에서는 요소 수로 어림하고, 이후에는 앞선 분석의 출력 길이로 계산합니다. HTML 페이지에는 진행 막대와 "분석 취소" 버튼이, ComfyUI 노드에는 노드 진행 막대와 콘솔 표시가 나오고 ComfyUI 의 중단 버튼으로 멈출 수 있습니다. 파이썬에서는:

```python
import threading
stop = threading.Event()
result = an.analyze("photo.jpg", progress=lambda p: print(p), cancel=stop)   # p: Progress (stage, percent, input_tokens, output_tokens, tokens_per_sec, elapsed, eta)
print(result["stats"])
```

분석 지시문은 `promptmaker/data/vision/analyze_prompt.txt` 에서 고칠 수 있습니다. 요소 목록과 예시 문구는 현재 요소 파일에서 자동으로 채워지므로 요소를 새로 추가해도 따로 고칠 필요가 없습니다.

## 7-1. 로컬 LLM (선택, 0.5.0) - 요소 검증 · 요청 범위 검사

PromptMaker 의 기본 흐름은 그대로입니다: 한글 요청 → 번역 → 요소에서 검색 → **요청에 없는 요소는 관계 확률로 채움** → 영문 문법에 맞는 자연어 프롬프트.
`promptmaker.llm` 은 이 흐름을 바꾸지 않고, 두 곳에서만 짧게 확인합니다. 모델이 없으면 둘 다 꺼진 채로 동작합니다.

1. **요소 저장 전 검증** `llm.verify_items(items)` / `llm.verify_grouped({key: [text]})`
   - 요소 txt 에 넣기 전에 "이 문구가 이 요소에 맞는 짧은 단어/구인가?" 확인 → 그대로(`ok`) / 다른 요소로 이동(`move`) / 저장 안 함(`drop`)
   - 틀린 항목만 `번호=요소` 로 답하게 해서 출력은 보통 1~10 토큰
2. **생성 시 요청 범위 검사** `PromptMaker(scope_checker=llm.ScopeChecker())` + `GenerateOptions(scope_check=True)`
   - 요청이 정하지 않아 관계 확률로 채운 요소만 보내고, 요청과 **어긋나거나 범위를 벗어난** 것의 번호만 받음 (요청에 없던 요소를 채우는 것 자체는 정상)
   - 걸린 항목은 그 생성에서만 후보에서 빼고 관계 확률로 다시 고름 (최대 2번). 결과는 `result.lexicon["scope"]`

```python
from promptmaker import PromptMaker, llm

cfg = llm.LLMConfig(model_path=r"D:\models\Qwen3-4B-Instruct-2507")   # 폴더(config.json) 또는 .gguf
llm.save_config(cfg)                       # config/llm.json (PROMPTMAKER_LLM_CONFIG 로 위치 변경)
pm = PromptMaker(scope_checker=llm.ScopeChecker(cfg))
pm.generate("침실에서 책 읽는 여자", scope_check=True)
llm.verify_items([{"key": "top", "text": "a red string bikini"}], cfg=cfg)   # → swimwear 로 이동
llm.unload_llm()                           # VRAM 반환
```

- 백엔드 (0.6.0)
  - `vision`: **이미지 분석(VL) 모델을 함께 씀** (`config/vision.json` 그대로, 이미 올라가 있으면 재사용 · 추가 설치·VRAM 없음). `auto` 에서 모델 경로를 비우면 이것.
  - `transformers`: 모델 폴더(Qwen3 / Qwen2.5 Instruct, Qwen3-VL 폴더도 텍스트로 사용)
  - `llama_cpp`: .gguf (llama-cpp-python 필요 · Qwen3-VL 같은 새 구조는 최신 빌드 필요)
  - `llama_server`: .gguf 를 llama.cpp 공식 `llama-server(.exe)` 로 자동 실행 (`server_path`, `server_port`) · 설치 없이 최신 모델 지원
  - `openai`: Ollama/LM Studio 등 서버
  - `auto`: 경로를 보고 선택. **GGUF 는 백엔드를 무엇으로 골랐든 transformers 로 열지 않고** llama-cpp-python(있으면) → llama-server 순서로 씀
- 속도: 탐욕 디코딩, Qwen3 생각(thinking) 모드 끔, 출력 상한을 항목 수에 맞춰 작게
- 성인 콘텐츠는 일반 문구처럼 판단합니다(거절·검열 없음). 미성년 연상 문구는 그 전에 `safety` 규칙으로 제외됩니다.
- 시험: `python -m promptmaker.llm "top=a red string bikini" --model D:\models\Qwen3-4B-Instruct-2507`

### 7-2. 생성 충돌 규칙 (0.6.1 · LLM 없이 동작)

- **누드 요청**(nude, naked, 누드, 알몸, 나체, 전라 …)이면 `attire` 는 `nothing at all` 류, `attire_state` 는 `fully nude` 로 채우고 의상·신발·액세서리를 고르지 않습니다. `nude lipstick`, `semi-nude` 같은 표현은 누드 요청으로 보지 않습니다.
- 옷을 입은 장면에서는 `nude/naked/topless` 가 들어간 포즈·동작 항목을 고르지 않습니다.
- 인물 요소가 아닌 파일에 들어간 주어 문장 조각(`the woman, with dark hair`, `she is ...`)은 불러올 때·학습 저장 때 제외됩니다.
- 한 프롬프트에서 `bokeh`, `glow`, `golden hour` 같은 특징 단어가 이미 쓰였으면 다른 요소에서 그 단어가 든 항목의 확률을 1/10 로 낮춥니다.
- LLM 범위 검사(`scope_check`)도 '누드 요청인데 옷' 과 '깨진 조각'을 함께 잡도록 지시가 보강됐습니다.

### 7-3. 국적 · 여성 나이 규칙 (0.7.0)

- 기본값: 국적 **Korean · Japanese**, 여성 나이 **20~22세**. 패키지의 `nationality.txt` · `age.txt` 도 이 값만 들어 있습니다.
- 바꾸려면 **사용자 데이터 폴더**의 `restrictions.json` 에 저장하세요 (패키지 파일을 직접 고치면 업데이트 때 덮어써집니다).

```python
from promptmaker import PromptMaker, Restrictions
Restrictions.from_dict({"nationalities": ["Korean", "Japanese"], "female_age_min": 20, "female_age_max": 22}).save(user_dir)
pm = PromptMaker(user_data_dir=user_dir)            # 또는 PromptMaker(restrictions={...}), {"enabled": False} 로 끔
```

- 생성: 후보를 규칙 안 값으로만 고르고, 요청이 규칙 밖(예: 30살 중국 여성)이면 규칙 안으로 바꾸고 경고를 남깁니다. 남성은 그대로.
- 이미지 분석 · 학습 저장(`append_items`, `classify`): 규칙 밖 국적/나이는 저장하지 않음 (`restricted`). 분석 풀프롬프트도 `apply_to_text` 로 바꿉니다.
- 기존 파일 정리: `promptmaker.restrictions.clean_element_files(user_dir)` — 규칙 밖 줄을 지우기 전에 `backup/restrictions-<시각>/` 에 백업.
- 20세 미만 금지 안전 규칙은 항상 먼저 적용됩니다 (규칙 최소값도 20 미만으로 내려가지 않음).

### 7-5. 샷 구도 · 탈의 장면 (0.9.0 · LLM 없이 동작)

**샷 구도**는 동작·포즈와 따로 정하고, 문장 하나로 주제 문장 바로 뒤에 넣습니다 (템플릿에 `{framing}` 자리가 있으면 그 자리).

| 구도 id | 요청 예 (한글/영문) | 담기는 범위 |
|---|---|---|
| `extreme_closeup` | 얼빡샷, 얼굴만, 초근접, extreme close-up, face only | 얼굴만 (의상·포즈 생략, '뷰티 클로즈업' 템플릿) |
| `closeup` | 클로즈업, 헤드샷, close-up, headshot | 머리 ~ 어깨 |
| `medium` | 미디엄샷, 상반신, 바스트샷, medium shot, waist up | 허리 위 |
| `cowboy` | 카우보이샷, 무릎 위, cowboy shot | 허벅지 중간 위 |
| `full_body` | 전신샷, 풀샷, full body, full-length | 머리 ~ 발끝 |

- 구도에 안 담기는 포즈·동작·의상 상태·조명 효과·카메라 움직임은 고르지 않습니다 (미디엄샷에 '하이힐', '다리를 꼬고', 클로즈업에 '핸드백'). 의상도 하의·신발은 담기는 구도에서만 조합합니다. 단, 요청에 직접 쓴 의상은 남깁니다 (얼빡샷 제외).
- 카메라·사진 스타일 줄이 다른 구도를 말하면 (`full-body shot`, `studio beauty headshot`, `wide-angle environmental portrait`) 고르지 않습니다. 구도를 정하지 않았을 때 랜덤으로 고른 카메라가 구도를 말하면 (`medium shot from the waist up`) 나머지를 그 구도에 맞춥니다.
- 구도가 함께 쓸 수 없게 여러 개면 (`클로즈업 전신샷`) 마지막에 말한 하나만 쓰고 경고를 남깁니다. 얼빡샷 + 클로즈업처럼 둘 다 가까우면 더 가까운 쪽.
- '얼빡샷' 같은 한국어 구도 말은 번역 전에 영어로 바꿉니다 (구글 번역이 'a quick shot' 으로 옮기는 문제).
- 구도 문구는 `elements/framing.txt` 에서 고칠 수 있습니다. 줄 끝 `#extreme_closeup` 같은 태그가 구도를 정합니다.

**탈의 장면**('옷을 벗는 중', '옷을 벗는 장면', '블라우스 단추를 푸는', 'undressing', 'taking off her top')은 누드(이미 벗은 상태)와 다릅니다.

- 옷은 평소처럼 조합하고, 동작 자리(`action` → `motion` → `pose` 순, 없으면 문장 추가)에 탈의 동작을, 의상 상태 자리에 탈의 중 상태를 넣습니다.
- 탈의 동작은 `elements/undressing.txt`, 의상 상태는 `elements/undressing_state.txt` (편집 가능). 줄 안의 `{top}` `{bottom}` `{dress}` `{swimwear}` `{lingerie}` `{sleepwear}` `{sportswear}` `{traditional}` `{outerwear}` `{garment}` 은 지금 입은 옷 이름으로 바뀌고 ('a School-style tennis micro mini pleated skirt' → 'pleated skirt'), 입지 않은 옷을 말하는 줄은 쓰지 않습니다.
- 요청에 옷 이름·동작이 있으면 그 줄을 우선합니다 ('블라우스를 벗는' → 블라우스, '단추를 푸는' → unbuttoning). 단추 없는 옷에 'unbuttoning', 지퍼 없는 옷에 'unzipping' 은 쓰지 않습니다.
- 손이 바쁘거나 몸 전체가 움직이는 포즈·동작(책을 든, 춤추는, 걷는 …)은 함께 고르지 않습니다.
- 샷 구도와 함께: 구도에 안 담기는 탈의 동작은 빠집니다 (미디엄샷에 'down over her hips' X). 얼빡샷이면 탈의 동작을 넣지 않고 경고를 남깁니다.
- '탈의실', '머리를 푸는' 은 탈의 요청이 아닙니다.

### 7-6. 신체 부위별 클로즈업 · 분석 구도 분리 (0.11.0)

요청에 **부위 + 클로즈업 말**이 붙어 있으면 부위 클로즈업으로 봅니다: '다리 클로즈업', '입술 접사', '손만 나오게', '클로즈업된 허벅지', 'close-up of her legs', 'lips close-up'. ('다리 위에 서 있는', '목걸이' 처럼 클로즈업 말이 없으면 아님)

| 부위 id | 한국어 | 보이는 의상 |
|---|---|---|
| `eyes` `lips` `hair` | 눈·속눈썹, 입술, 머리카락 | 없음 (메이크업·헤어로 꾸밈) |
| `hands` | 손·손가락·손톱 | 반지·팔찌 |
| `neck` `shoulders` `chest` `back` | 목선·쇄골, 어깨, 가슴·데콜테, 등·뒤태 | 상의·원피스·수영복 등 |
| `waist` `hips` | 허리·배꼽·복근, 엉덩이·골반 | 상·하의, 수영복 등 |
| `thighs` `legs` | 허벅지, 다리·각선미 | 하의·스타킹 (다리는 신발도) |
| `feet` | 발·발가락·발목 | 신발·스타킹·발찌 |

- 전용 템플릿 `08_body_part_closeup.json` (`framing_only`: 부위 클로즈업일 때만 쓰임) 으로 만들고, 다른 템플릿을 고르면 구도 문장만 넣습니다.
- 포즈·동작은 그 부위를 말하는 것만 씁니다 (다리 → 'sitting with legs crossed', 얼굴 표정 X). 맞는 게 없으면 그 문장을 빼요. 'lying on her back' 은 등 클로즈업이 아닙니다.
- 구도 문구는 `elements/body_closeup.txt` (줄 끝 `#legs` 같은 태그, 없으면 문구로 부위 판단).
- 웹 테스트 페이지·ComfyUI 노드의 '신체 부위 클로즈업' 선택, 라이브러리 `body_part="legs"`.

**이미지 분석 구도 분리**: 분석 결과의 카메라/구도 문구는 저장할 때 나눕니다.
- 'close-up of her legs' → `body_closeup.txt` 에 `close-up of her legs #legs`
- 'medium shot from the waist up' → `framing.txt` 에 `... #medium` (렌즈·각도가 들어간 줄은 카메라에 그대로)
- `framing` / `body_closeup` 에 구도가 아닌 문구가 오면 저장하지 않음 (`wrong_key`)
- `undressing` / `undressing_state` (문구 틀) 는 분석 대상에서 뺌

### 7-7. 와일드카드 확장 요소 묶음 (0.12.0)

AI Gallery Manager 의 일반 와일드카드(마스터 제외) 프롬프트 약 4만 5천 줄을 요소별로 나눈 문구 약 2만 1천 줄이
`promptmaker/data/packs/wildcard/elements/<요소>.txt` 에 들어 있습니다. 모두 요소 검사(7-4)를 통과했고, 기본 요소와 겹치지
않으며, 국적(Korean·Japanese 외)·나이·남성 묘사가 든 문구는 없습니다.

기본 요소와 달리 장면 태그(`#beach`, `#winter` …)가 없어서 **기본은 꺼져 있습니다** (켜면 해변 장면에 침실 소품처럼 장면이
어긋날 수 있음). 필요할 때 켭니다.

```python
from promptmaker import PromptMaker, available_packs
available_packs()                        # ['wildcard']
pm = PromptMaker(packs=["wildcard"])     # 또는 환경변수 PROMPTMAKER_PACKS=wildcard
```

읽는 순서는 기본 요소 → 확장 묶음 → 사용자 데이터 폴더입니다.

### 7-8. 자위 장면 · 요청 내용 그대로 반영 (0.13.0)

**자위 장면** (성인 여성 혼자): 요청에 '자위', '스스로 만지는', 'masturbating', 'touching herself' 가 있으면
`elements/masturbation.txt` (사용자 편집 가능, 현재분사구) 에서 동작 한 줄을 넣습니다.

| 요청 말 | 우선하는 동작 |
|---|---|
| 가슴 · 유두 / breast · nipple | 가슴 주무르기 (`fondling {her} own breasts with one hand` …) |
| 클리토리스 · 음핵 / clit | 클리토리스 만지기 (`rubbing {her} clitoris with {her} fingertips` …) |
| 손가락 · 질 / finger inside · vagina | 질에 손가락 넣기 (`sliding two fingers inside {her} vagina` …) |

- 동작 자리(action/motion)가 비어 있으면 거기에, 탈의 동작이 차지했으면 따로 한 문장으로 넣습니다.
- 포즈는 눕기·앉기·무릎 꿇기 같은 것만 (서 있기·걷기 X), 손이 바쁜 포즈·동작 X, 신발 X, 쾌감 표정 우선.
- 장소를 말하지 않으면 침대·소파·방 같은 실내 사적인 곳을 우선합니다.
- '팬티를 내리며' → 탈의 동작이 팬티를 내리는 줄로 (`undressing.txt` 의 팬티 줄은 요청에 팬티·속옷이 있을 때만 씀).

**요청 내용 그대로 반영** (AI채팅 오생성 사례 '긴생머리 … 미니슬립 … 침대에서 … 자위'):

- 번역 전 처리: '긴생머리' → long straight hair, '미니슬립 의상' → mini slip dress (`translator.KO_PRE_TERMS`),
  번역문 'mini-slip costume' → mini slip dress (`normalize_english`).
- 머리 수식어(`long straight hair`)는 머리 요소에서만, 색·길이 같은 수식어 하나만으로는 의상·포즈를 잠그지 않음
  ('long' 이 긴 치마·긴팔 상의를 잠그던 문제), 'hair' 는 머리 요소에서만 찾음.
- 장소는 중심 명사가 같은 줄만 ('bed' → 'on a bed with white sheets' O, 'on a gravel garden bed'·'on a telephone beside bed' X).
  요소에 그 장소가 없으면 요청한 말 그대로 ('on a sofa').
- 'seductive' 같은 외모 형용사는 외모(beauty) 요소로 (포즈 'standing in a seductive' 를 잠그지 않음).
- 'costume', 'outfit', 'high-definition' 은 'The scene also features …' 에 남지 않음.
- 의상 종류는 요청 문장에 실제 옷 이름이 있는 종류를 우선 ('white bikini' 요청에 후디가 걸려도 수영복).
- 요소 검사(7-4) 추가: '… and her long / and her hair' 꼬리는 잘라 냄, '… with a seductive' 처럼 끝이 잘린 줄,
  영어가 아닌 글자(중국어 등)가 섞인 줄은 쓰지 않음. 와일드카드 묶음(7-7)도 같은 규칙으로 다시 정리.

### 7-9. 피사체 거리 · 비율 추천 (0.14.0)

**피사체 거리** (`promptmaker/distance.py`, `elements/distance.txt`): 카메라가 피사체에서 얼마나 떨어져 찍는지.
샷 구도(7-5)가 '몸의 어디까지 담는가' 라면, 거리는 '얼마나 떨어져서 찍는가' 입니다.

| 거리 id | 요청 말 | 어울리는 구도 |
|---|---|---|
| `close` 근거리 | 가까이서, 근거리, 코앞 / up close, close range | 얼빡샷 · 클로즈업 · 미디엄 · 부위 클로즈업 |
| `medium` 중거리 | 중거리, 몇 걸음 떨어져 / from a few steps away | 클로즈업 ~ 전신 |
| `far` 원거리 | 멀리서, 원거리, 롱샷 / from a distance, from afar | 카우보이 · 전신 |
| `very_far` 원경 | 아주 멀리서, 원경 / extreme long shot, tiny figure | 전신 |

- `GenerateOptions(distance="auto")` (기본: 요청에서 찾음) · `close` · `medium` · `far` · `very_far` · `none`(넣지 않음).
  요소 목록에서 `distance` 문구를 직접 골라도 됩니다 (`selections={"distance": "..."}`).
- 문구는 구도 문장 바로 뒤에 한 문장으로 들어갑니다 (`picks["distance"]`).
- 구도와 맞지 않으면 (얼빡샷 + 멀리서) 구도를 따르고 거리를 구도에 맞는 쪽으로 바꾸며 경고를 남깁니다.
- 구도를 말하지 않았으면 거리로 구도를 정합니다 (멀리서 · 원경 → 전신샷, 가까이서 → 미디엄샷).
- 멀리서 찍는 장면에는 '셀카' · '얼굴 가득' 같은 가까운 줄을 고르지 않습니다.
- 요청에 거리가 없으면 무작위 모드에서 구도가 정해진 장면에 가끔(30%) 구도에 맞는 거리 문구를 넣습니다.
- '가까이 다가가는' 처럼 동작인 말은 거리로 보지 않고, 'into the distance' (먼 곳을 바라봄) 도 거리가 아닙니다.
- 요소 검사(7-4): 'shot from a distance' 처럼 거리만 말하는 줄은 `distance` 에 `#far` 같은 태그를 붙여 옮깁니다.

**비율 추천** (`promptmaker/aspect.py`): 완성된 프롬프트의 핵심 피사체로 화면 비율을 정합니다.

```python
from promptmaker import suggest_aspect
suggest_aspect("Full-body shot showing her from head to toe. She is lying on her side on a bed.")
# {'ratio': '16:9', 'orientation': 'landscape', 'reasons': ['누운 · 기댄 자세 → 가로', ...], ...}
```

- 자세 · 포즈 · 행동: 서 있기 · 걷기 → 세로 / 눕기 · 엎드리기 → 가로 / 앉기 · 무릎 꿇기 → 약한 세로 / 양반다리 · 웅크리기 → 정사각형.
  피사체(she · her · woman) 문장에서 처음 나온 자세를 크게 봅니다.
- 구도: 얼빡샷 → 정사각형, 클로즈업 · 미디엄 · 카우보이 · 전신 → 세로. 거리: 원거리 → 가로, 원경 → 넓은 가로(2:1).
- 장면: 와이드 · 파노라마 · 풍경 → 가로, 두 사람 이상 → 가로.
- 결과 `ratio` 는 `9:16` · `1:1` · `16:9` · `2:1` · `9:18` 중 하나. 판단할 것이 없으면 `9:16`.

### 7-10. 카메라 시점 · 앵글 (0.16.0)

**카메라 시점** (`promptmaker/view.py`, `elements/view.txt`): 카메라가 피사체를 **어느 쪽에서, 어느 높이에서** 보는지.
샷 구도(7-5) · 피사체 거리(7-9)와 따로 정해집니다. 실사 인물 카메라 컨트롤 LoRA 사례(42개 시점 학습 캡션)의 방식을 따릅니다.

| 축 | id · 요청 말 |
|---|---|
| 방향 (`view`) | `front` 정면 · `front_3q` 정면 사선(사선, 반측면, 3/4, three-quarter view) · `side` 측면(옆모습, side profile) · `rear_3q` 뒤 사선(뒤돌아보는, looking back over her shoulder) · `rear` 뒷모습(뒤태, from behind, rear view) |
| 높이 (`view_height`) | `eye_level` 눈높이 · `high` 하이앵글(내려다보는 시점, 부감) · `overhead` 탑뷰(바로 위에서, top-down) · `low` 로우앵글(올려다보는 시점, 앙각) · `worm` 웜즈아이(바닥에서 올려다보는) |
| 기울기 · 화면 방향 | 더치 앵글 (dutch angle) · 화면 왼쪽/오른쪽을 향함 (facing screen left/right) — 요청에서만 |

- `GenerateOptions(view="auto", view_height="auto")` (기본: 요청에서 찾음), 값을 직접 주거나 `none`(넣지 않음).
- 문구는 구도 · 거리 문장 바로 뒤에 한 문장으로 들어갑니다 (`picks["view"]`, 템플릿에서는 `{view}`).
- **몸통이 어떻게 보이는지 함께 적습니다**: 'front three-quarter view, … with the front and one side of her torso visible'.
  정면 사선은 이름만 쓰면 측면처럼 나오는 경우가 많았습니다. `view.txt` 에 줄을 추가할 때도 같은 방식으로 적으세요.
- **뒷모습**: 표정 자리는 '얼굴을 돌린' 말로 바꾸고 (요청에 표정을 말했으면 그대로), 카메라와 눈을 맞추는 · 윙크하는 줄은 빼고,
  눈 화장은 드물게만 고릅니다. 템플릿의 '카메라를 바라보는' 문장은 'seen from behind' 로 바꿉니다.
- **뒷모습 · 뒤 사선**: '카메라 쪽으로 걸어오는' 은 '카메라에서 멀어지는' 으로 바꾸고, '카메라를 정면으로 보는' 줄은 뺍니다.
- 시점 문구를 넣으면 각도를 말하는 카메라 줄('low-angle shot', 'Dutch angle composition')은 고르지 않습니다 (렌즈 · 구도만).
- 바로 아래에서 수직으로 올려다보는 시점은 인물 형태가 무너지므로 웜즈아이는 '바닥 가까이에서 올려다보는' 으로 씁니다.
- 요청에 시점이 없으면 무작위 모드에서 가끔(25%) 정면 사선 · 측면을 넣고, 그중 30% 는 약한 하이 · 로우앵글을 더합니다
  (뒷모습 · 탑뷰 · 웜즈아이 · 더치 같은 극단 시점은 무작위로 넣지 않음).
- '뒤에서 안는 (hugging her from behind)' 같은 상호작용, 'light from above' 같은 조명, 'Dutch woman' 은 시점이 아닙니다.
- 요소 검사(7-4): 이미지 분석이 카메라에 넣은 'shot from a low angle' 처럼 시점만 말하는 줄은 `view` 에 `#low` 같은 태그를 붙여 옮깁니다.

### 7-11. 프롬프트 커스터마이징 (0.18.0)

캐릭터 커스터마이징 화면처럼 **피사체 1·2 → 장소/배경 → 조명 세팅 → 앵글** 순서로 요소를 골라 프롬프트를 만듭니다.
(AI Gallery Manager 의 AI채팅 '커스텀프롬' 화면이 이 기능을 씁니다.)

```python
from promptmaker import PromptMaker
from promptmaker.customize import schema, build, build_batch

pm = PromptMaker()
info = schema(pm)      # 단계(steps) · 카테고리(categories, 성별별 카드 items) · 시점 이름
res = build(pm, {
    "subjects": [
        {"enabled": True, "gender": "female",
         "sel": {"hair": {"mode": "pick", "values": ["long wavy chestnut hair"]},
                 "outfit": {"mode": "pick", "values": ["a satin camisole", "a pleated tennis skirt"]},
                 "action": {"mode": "off"}}},
        {"enabled": True, "gender": "male"},          # 두 번째 피사체 (생략하면 한 명)
    ],
    "scene": {"location": {"mode": "auto"}, "view": {"direction": "front_3q", "height": "high"}},
}, model_profile="krea2")
print(res["prompt"])
```

- 카테고리마다 `mode`: `auto`(요소 txt 에서 자동 선택, 이미 고른 장소·의상과 어울리는 항목 우선) · `pick`(고른 값) · `off`(넣지 않음)
- 피사체: 성별 · 국적 · 나이 · 외모 수준 / 헤어스타일 · 머리 길이 · 메이크업 / 의상(여러 개) · 디자인 · 소재 · 착용 상태 / 포즈 / 동작 · 표정
- 장소: location · time_weather, 조명: lighting · lighting_effect · color_grade · photo_style, 앵글: view(다이얼 값) · framing · camera
- 새 요소 txt: `hair_length.txt`(머리 길이) · `outfit_design.txt`(디자인, with 구) · `outfit_material.txt`(소재, in 구) · `male_hair.txt`(남성 헤어)
- 국적 · 여성 나이 규칙(7-3)을 그대로 지킵니다. 남성 나이는 23~35 중에서 고릅니다.
- 디자인 · 소재 · 착용 상태 · 조명 효과 · 색감 같은 꾸밈 요소는 자동일 때 가끔만 넣습니다 (`auto_rate`).
- 뒷모습 시점이면 표정은 자동으로 넣지 않고, '카메라 쪽으로' 는 '카메라에서 멀어지는' 으로 바꿉니다.
- 웹 서버(`run_server.bat`): `GET /api/customize/schema`, `POST /api/customize/build` (`{"spec": {...}, "count": 1}`)

### 7-4. 요소 검사 · 잘못 분류된 요소 정리 (0.8.0)

요소 txt 한 줄은 **단어 또는 짧은 구** 여야 합니다 (템플릿이 요소를 엮어 문장을 만듦). 이미지 분석·생성 프롬프트 자동 추가가 긴 문장을 쉼표로만 잘라 넣으면 이런 줄이 생겼습니다.

| 원래 파일 | 줄 | 문제 |
|---|---|---|
| camera | `19age is visible in a close-up portrait shot` | 주어 + 동사 문장 |
| camera | `her dark black hair styled with straight bangs framing her face` | 헤어인데 `framing` 때문에 카메라 |
| lighting | `a light blue flower pinned near {her} neck` | `light` 가 색 이름 |
| color_grade | `a muted pink lipstick` | 메이크업인데 `muted` 때문에 색감 |
| action | `suggesting an indoor setting with ambient lighting` | 장면 설명 분사구 |
| pose | `looking directly at the camera with a gentle` | 끝이 잘린 조각 |

`promptmaker.element_check` 가 한 줄을 판단합니다 (규칙 + 사전, AI 미사용, 수위 제한 없음).

```python
from promptmaker.element_check import check_item
check_item("camera", "dark hair framing her face")
# {"status": "moved", "key": "hair", "text": "dark hair framing her face", "reason": "..."}
```

| status | 뜻 | 저장(`append_items`) | 불러오기·생성 |
|---|---|---|---|
| `ok` | 그대로 사용 | 저장 | 사용 |
| `fixed` | 끝이 잘린 문장을 다듬음 (`bathed in soft light that` → `soft light`) | 다듬은 문구 저장 | 다듬은 문구 사용 |
| `moved` | 다른 요소에 맞음 | 맞는 요소 파일에 저장 | 맞는 요소로 옮겨 사용 |
| `fragment` | 문장 조각 (주어·동사 문장, 나이·키 표기, '없음' 문장, 추측 표현, 14단어 넘는 문장) | 건너뜀 | 쓰지 않음 |
| `wrong_key` | 형식이 맞지 않고 맞는 요소도 못 찾음 | 건너뜀 | 쓰지 않음 |

- 적용 위치: `append_items()`(이미지 분석·학습 저장, `check=False` 로 끌 수 있음), `classify()`(분석 결과 표시: 행에 `key` · `original` 이 붙음), `load_elements()`(생성, 파일은 바꾸지 않음, `check=False` 로 끔).
- 판단 대상: 헤어·메이크업·피부·표정·포즈·동작·의상류·장소·시간·조명·카메라·색감·사진 스타일·분위기·외모. 영상·상호작용·의상 상태 등은 판단하지 않습니다.
- 기본 요소(패키지 데이터)는 모두 `ok` 여야 합니다 (`tests/test_element_check.py`). 사전을 고칠 때 이 테스트로 확인하세요.
- 0.10.0: 와일드카드 프롬프트(약 4만 5천 줄)를 요소로 나눠 보며 찾은 잘못된 줄도 거릅니다 — 추상 표현·옷 입기 동사로 시작하는 동작(`embodying …`, `rocking a mesh bodysuit`), 헤어에 들어간 옷(`a fringe chemise` → 속옷), 장소가 아닌 `on a palm trees …`, 장면 묘사가 붙은 조명(`pedestrians under bright daylight sky`), 몸매 뜻의 `toned`·`wood grain` 색감, `technical details : …` 같은 라벨, `woman`·`micro-pores` 같은 피사체·피부결 묘사, 끝에 형용사·색만 남은 조각(`holding a bouquet of pink`).
- 이미 쌓인 파일 정리 (먼저 `<사용자 폴더>/backups/elements_<시각>/` 에 백업, `wrong_key` 줄은 그대로 둠):

```bash
python -m promptmaker.element_check <사용자 데이터 폴더>           # 미리보기
python -m promptmaker.element_check <사용자 데이터 폴더> --apply   # 백업 후 적용
```

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
python tools/github_auto.py release check|prepare|publish   # 버전 관리 (9-1)
```

`auto` 는 main 에서 실행하면 직접 푸시하지 않고 `--branch` 로 새 브랜치를 만들도록 요구합니다.

### 9-1. 버전 관리 (규칙)

다른 프로젝트에서 GitHub 버전으로 이 라이브러리를 가져다 쓰므로, **main 에 병합할 때마다 버전을 올리고 태그 + GitHub Release 를 만듭니다.**

1. 작업 브랜치(PR)에서 버전 올리기: `python tools/github_auto.py release prepare` (기본 minor, `--bump patch|major` 또는 `--version 0.3.0`). `pyproject.toml` 과 `promptmaker/__init__.py` 의 버전을 함께 바꾸고 커밋합니다.
2. PR 을 main 에 병합합니다.
3. main 에서 `git pull` 후 `python tools/github_auto.py release publish` → `v<버전>` 태그를 푸시하고, 지난 태그 이후 병합된 PR 제목으로 GitHub Release 를 만듭니다.
4. 확인: `python tools/github_auto.py release check`

버전 규칙: 기능 추가는 minor (0.3.0 → 0.4.0), 버그 수정만이면 patch (0.3.0 → 0.3.1).

**다른 곳에서 버전 확인 / 설치**

```bash
pip install git+https://github.com/wowjjang83-sketch/PromptMaker.git@v0.3.0   # 특정 버전
python -m promptmaker --version
python -m promptmaker --check-update      # GitHub 최신 릴리스와 비교
```

```python
import promptmaker
from promptmaker.version import check_for_update, latest_release
print(promptmaker.__version__)
print(check_for_update())   # {"current", "latest", "update_available", "url", "install"}
```

## 10. 문제 해결

| 증상 | 해결 |
|---|---|
| 경고 "구글 번역 실패" | 네트워크/요청 제한. `pip install deep-translator` 또는 영어로 입력. 오프라인 사전으로 키워드는 계속 동작합니다 |
| 수정한 데이터가 반영 안 됨 | `pm.reload()`, HTML "데이터 다시 읽기", ComfyUI "데이터 다시 읽기" 노드 |
| 결과에 `{...}` 가 남음 | 슬롯 이름 오타. 해당 이름의 요소 파일이 있는지 확인 |
| 특정 문장이 빠짐 | 슬롯 요소가 비어 있거나 `when` 조건이 거짓 |
| 문법이 어색함 | 항목의 문법 형태(5-1 표)를 맞추고, she/her 대신 `{she}` `{her}` 사용 |
