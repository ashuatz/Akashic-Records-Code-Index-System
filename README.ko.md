# Akashic Records Code Index System (한국어)

영문 문서: `README.md`

## 이 도구가 하는 일

이 프로젝트는 AI 어시스턴트가 사용할 수 있는 MCP 기반 코드 검색 서비스입니다.

핵심 목적:
- 대규모 코드베이스를 키워드가 아니라 “의도” 중심으로 검색
- 필요한 코드 문맥만 선택적으로 조회
- 심볼/정의/참조 탐색을 도구 호출로 일관되게 수행

즉, 프롬프트 컨텍스트 낭비를 줄이고 답변 정확도와 재현성을 높이기 위한 도구입니다.  
이 문서에는 Unity 소스 내부 구현 등 민감한 상세 내용은 포함하지 않습니다.

## 빠른 시작

1. 환경 파일 생성
```bash
cp .env.example .env
```

2. 서비스 기동
```bash
docker compose up -d --build
```

3. 상태 확인
```bash
docker compose ps
curl http://localhost:8088/health
```

Windows 단축 스크립트:
- `start_all.bat`
- `stop_all.bat`
- `status.bat`

## 주요 엔드포인트

- MCP HTTP: `http://localhost:8088`
- Health: `http://localhost:8088/health`
- Qdrant: `http://localhost:6333/collections`
- 선택 MCP 라우트: `GET /mcp/sse`, `POST /mcp/message`

## 스킬 구성 방법 (Claude Code)

Claude Code용 스킬은 아래처럼 최소 구조로 구성하면 됩니다.

```text
~/.claude/skills/akashic-mcp/
├── SKILL.md
└── references/
    └── api-reference.md   # 선택
```

`SKILL.md` 필수 요소:
- `name`: 소문자-하이픈 형식 (예: `akashic-mcp`)
- `description`: 어떤 질문에서 스킬을 트리거할지 명시
- 워크플로우: 어떤 툴을 어떤 순서로 호출할지 안내

트리거 예시:
- “X가 어디 구현됐는지 찾아줘”
- “X 참조 위치 찾아줘”
- “X 주변 코드 컨텍스트 보여줘”
- “X 정의로 이동해줘”

권장 도구 흐름:
1. `search_code`로 후보 탐색
2. `get_symbol` / `get_cpp_symbols`로 대상 확정
3. `get_file_context`로 정밀 문맥 확인
4. `find_cpp_references` / `go_to_definition`으로 추적

스킬 작성 시 주의:
- 반복적으로 필요한 절차만 간결하게 작성
- 긴 API 설명은 `references/*`로 분리
- 저장소 비공개 정보/민감한 구현 디테일은 넣지 않기

## 설정 문서

- `Docs/Deployment-Guide.md`
- `Docs/Configuration-Reference.md`
- `.env.example`
