# PDF RAG 시스템

Ollama를 사용한 로컬 PDF 기반 질의응답 시스템입니다. LangChain과 ChromaDB를 활용하여 PDF 문서를 벡터화하고, LLaMA 3 모델로 한국어 답변을 제공합니다.

## 🚀 주요 기능

- **로컬 LLM 사용**: Ollama를 통한 완전한 로컬 실행
- **PDF 문서 처리**: PyMuPDF를 사용한 안정적인 PDF 텍스트 추출
- **벡터 검색**: ChromaDB를 활용한 의미 기반 문서 검색
- **스마트 캐싱**: 파일 해시 기반 벡터 저장소 재사용으로 성능 최적화
- **웹 인터페이스**: Gradio를 통한 직관적인 사용자 인터페이스
- **설정 파일 지원**: YAML 기반 설정 관리

## 📋 요구사항

- Python 3.8 이상
- Ollama 설치 및 실행 중
- 다음 Ollama 모델이 설치되어 있어야 합니다:
  - `llama3` (LLM 모델)
  - `mxbai-embed-large` (임베딩 모델)

## 🔧 설치 방법

### 1. 저장소 클론 또는 다운로드

```bash
git clone <repository-url>
cd PDF_RAG
```

### 2. 가상환경 생성 및 활성화 (권장)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. Ollama 모델 설치

```bash
# LLM 모델 설치
ollama pull llama3

# 임베딩 모델 설치
ollama pull mxbai-embed-large
```

### 5. Ollama 서버 실행 확인

Ollama가 실행 중인지 확인하세요. 실행되지 않은 경우:

```bash
ollama serve
```

## ⚙️ 설정

`config.yaml` 파일을 통해 시스템 설정을 변경할 수 있습니다:

```yaml
ollama:
  llm_model: "llama3"              # 사용할 LLM 모델명
  embedding_model: "mxbai-embed-large"  # 사용할 임베딩 모델명

text_splitter:
  chunk_size: 1000                 # 청크 크기
  chunk_overlap: 200               # 청크 간 겹치는 문자 수

vectorstore:
  cache_directory: "chroma_pdf_cache"  # ChromaDB 캐시 디렉토리
  collection_name: "pdf_documents"     # 컬렉션 이름

system_prompt: "..."               # 시스템 프롬프트
gradio:
  title: "..."                     # Gradio 인터페이스 제목
  description: "..."               # Gradio 인터페이스 설명
```

## 🎯 사용 방법

### 1. 애플리케이션 실행

```bash
python Ollama_PDF_RAG.py
```

### 2. 웹 인터페이스 사용

1. 브라우저에서 표시된 URL로 접속 (기본: http://127.0.0.1:7860)
2. PDF 파일 업로드
3. 질문 입력
4. 답변 확인

### 3. 성능 최적화

- **첫 번째 질문**: PDF를 로드하고 벡터화하는 과정이 필요하므로 시간이 걸릴 수 있습니다.
- **이후 질문**: 동일한 PDF 파일에 대한 질문은 캐시된 벡터 저장소를 재사용하므로 빠르게 처리됩니다.

## 📁 프로젝트 구조

```
PDF_RAG/
├── Ollama_PDF_RAG.py          # 메인 애플리케이션
├── config.yaml                 # 설정 파일
├── requirements.txt            # Python 의존성
├── README.md                   # 프로젝트 문서
├── chroma_pdf_cache/          # ChromaDB 벡터 저장소 캐시
│   ├── chroma.sqlite3         # SQLite 메타데이터
│   └── [collection_hash]/     # 벡터 데이터 파일들
└── [PDF 파일들]                # 테스트용 PDF 문서들
```

## 🔍 주요 개선 사항

### v2.0 (현재 버전)

- ✅ **성능 개선**: 파일 해시 기반 벡터 저장소 캐싱으로 동일 PDF 재사용
- ✅ **설정 파일 분리**: 하드코딩된 설정값을 `config.yaml`로 분리
- ✅ **에러 처리 강화**: 세부 예외 처리 및 사용자 친화적 메시지
- ✅ **코드 구조 개선**: 모듈화 및 가독성 향상

### v1.0 (이전 버전)

- 기본 RAG 기능
- Gradio 웹 인터페이스

## 🐛 문제 해결

### Ollama 연결 오류

```
❌ 연결 오류: Ollama 서버에 연결할 수 없습니다.
```

**해결 방법:**
1. Ollama가 실행 중인지 확인: `ollama list`
2. Ollama 서버 시작: `ollama serve`
3. 필요한 모델이 설치되어 있는지 확인: `ollama list`

### PDF 텍스트 추출 실패

```
❌ PDF에서 텍스트를 추출할 수 없습니다.
```

**해결 방법:**
1. PDF 파일이 손상되지 않았는지 확인
2. PDF가 이미지로만 구성된 경우 OCR이 필요할 수 있습니다
3. 다른 PDF 파일로 시도

### 벡터 저장소 오류

```
❌ 벡터 저장소 생성 중 오류 발생
```

**해결 방법:**
1. `chroma_pdf_cache` 디렉토리 권한 확인
2. 디스크 공간 확인
3. 캐시 디렉토리 삭제 후 재시도

## 📝 라이선스

이 프로젝트는 자유롭게 사용할 수 있습니다.

## 🤝 기여

버그 리포트나 기능 제안은 이슈로 등록해주세요.

## 📧 문의

프로젝트 관련 문의사항이 있으시면 이슈를 생성해주세요.

---

**Made with ❤️ using Ollama, LangChain, and Gradio**
