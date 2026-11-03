# 참고한 유사 프로젝트

| 프로젝트 | 방식 | PromptMaker 에 반영한 점 |
|---|---|---|
| [adieyal/sd-dynamic-prompts](https://github.com/adieyal/sd-dynamic-prompts) / [dynamicprompts](https://github.com/adieyal/dynamicprompts) | `{a\|b}` 변형 문법, `__wildcard__` TXT 파일, 가중치 `2::text` | 템플릿 안 `{a\|b}` 인라인 변형, TXT 한 줄 = 한 항목, `가중치::텍스트` 형식 |
| [adieyal/comfyui-dynamicprompts](https://awesome.ecosyste.ms/projects/github.com%2Fadieyal%2Fcomfyui-dynamicprompts) | 위 라이브러리의 ComfyUI 노드 | 시드 입력으로 재현 가능한 랜덤 생성 노드 구조 |
| [One Button Prompt](https://api.comfyonline.app/comfyui-nodes/One-Button-Prompt) ([A1111 버전](https://github.com/NotSoBot/nsb.webui.onebuttonprompt)) | 대량의 카테고리별 단어 목록 + 무작위 조합, 원 버튼 생성 | 카테고리별 대형 요소 목록, 전체/타입별 랜덤 모드 |
| [ComfyUI-Wildcard-Pipeline](https://github.com/DumiFlex/ComfyUI-Wildcard-Pipeline/wiki) | 와일드카드를 파이프라인으로 연결 | 요소 → 템플릿 → 후처리 단계 분리 |
| [comfyui-adaptiveprompts](https://www.runcomfy.com/comfyui-nodes/comfyui-adaptiveprompts) | 입력에 맞춰 와일드카드 선택 조정 | 사용자 입력 키워드와 일치하는 항목의 가중치 증가 |
| [comfyui-et_dynamicprompts](https://github.com/exectails/comfyui-et_dynamicprompts) | 경량 dynamic prompts 노드 | 외부 의존성 없는 구현 |
| Hugging Face `Gustavosta/MagicPrompt`, `succinctly/text2image-prompt-generator` | GPT-2 기반 프롬프트 확장 (LLM) | 추후 LLM 옵션을 붙일 수 있도록 `LLMRefiner` 훅만 마련 (기본은 사용 안 함) |

## 차별점

기존 와일드카드 도구는 대부분 **쉼표로 나열된 태그**를 만듭니다. PromptMaker 는

1. 문장 단위 템플릿과 문법 처리(관사 a/an, 단수/복수 she/they·is/are, 목록 연결 "A, B, and C")로 **자연어 문장**을 만들고,
2. 사용자 입력을 번역 → 키워드 검색 → 관련 단어를 확률적으로 뽑아 **입력 의도를 반영**하며,
3. 인물 수/성별 구성(casting)에 따라 문장 포함 여부(`when`)를 바꿉니다.
