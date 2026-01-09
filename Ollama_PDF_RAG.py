# Ollama_PDF_RAG.py
import gradio as gr
import ollama
import os
import hashlib
import yaml
from pathlib import Path
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

# 설정 파일 로드
def load_config():
    """설정 파일을 로드합니다."""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        # 기본 설정 반환
        return {
            'ollama': {
                'llm_model': 'llama3',
                'embedding_model': 'mxbai-embed-large'
            },
            'text_splitter': {
                'chunk_size': 1000,
                'chunk_overlap': 200
            },
            'vectorstore': {
                'cache_directory': 'chroma_pdf_cache',
                'collection_name': 'pdf_documents'
            },
            'pdfs_directory': 'pdfs',
            'system_prompt': 'You are a helpful assistant. Read the PDF content and answer the question. Translate the answer in Korean with emoji.',
            'gradio': {
                'title': 'LLaMA 3 - 다중 PDF 기반 질문 응답 시스템',
                'description': '여러 PDF 파일을 선택하고 질문을 입력하면, 모든 PDF의 내용을 기반으로 LLaMA 3가 한국어로 답변해 줍니다.',
                'output': {
                    'lines': 20,
                    'max_lines': 50,
                    'show_copy_button': True
                }
            }
        }

# 전역 설정 로드
CONFIG = load_config()

# 파일 해시 계산
def calculate_file_hash(file_path: str) -> str:
    """파일의 해시값을 계산합니다."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# 파일명에서 안전한 폴더명 생성
def get_safe_folder_name(file_path: str) -> str:
    """파일명에서 안전한 폴더명을 생성합니다."""
    file_name = Path(file_path).stem
    # 특수문자 제거 및 공백을 언더스코어로 변경
    safe_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in file_name)
    return safe_name[:50]  # 길이 제한

# 벡터 저장소 캐시 관리
_vectorstore_cache = {}  # {file_hash: vectorstore}

def get_or_create_vectorstore(file_path: str) -> Chroma:
    """파일 해시를 기반으로 벡터 저장소를 가져오거나 생성합니다."""
    try:
        file_hash = calculate_file_hash(file_path)
        file_name = Path(file_path).name
        safe_folder_name = get_safe_folder_name(file_path)
        
        # 캐시에 존재하는지 확인
        if file_hash in _vectorstore_cache:
            print(f"✅ 캐시에서 벡터 저장소를 재사용합니다. (파일: {file_name}, 해시: {file_hash[:8]}...)")
            return _vectorstore_cache[file_hash]
        
        # 벡터 저장소 디렉토리 경로 생성 (파일명_해시 기반)
        cache_dir = CONFIG['vectorstore']['cache_directory']
        # 각 PDF별로 별도 폴더 생성: [파일명]_[해시]
        collection_name = f"{safe_folder_name}_{file_hash[:8]}"
        vectorstore_path = os.path.join(cache_dir, collection_name)
        
        # 기존 벡터 저장소가 있는지 확인
        if os.path.exists(vectorstore_path) and os.path.isdir(vectorstore_path):
            print(f"✅ 기존 벡터 저장소를 로드합니다. (파일: {file_name}, 폴더: {collection_name})")
            embeddings = OllamaEmbeddings(model=CONFIG['ollama']['embedding_model'])
            vectorstore = Chroma(
                persist_directory=vectorstore_path,
                embedding_function=embeddings,
                collection_name=collection_name
            )
            _vectorstore_cache[file_hash] = vectorstore
            return vectorstore
        
        # 새로 생성
        print(f"📄 PDF 문서를 로드하고 벡터화합니다... (파일: {file_name}, 폴더: {collection_name})")
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        
        if not docs:
            raise ValueError(f"❗ PDF에서 텍스트를 추출할 수 없습니다: {file_name}")
        
        print(f"✅ PDF 문서 로드 완료. 파일: {file_name}, 총 {len(docs)} 페이지")
        if docs:
            print(f"   첫 페이지 미리보기: {docs[0].page_content[:200]}...\n")
        
        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG['text_splitter']['chunk_size'],
            chunk_overlap=CONFIG['text_splitter']['chunk_overlap']
        )
        splits = text_splitter.split_documents(docs)
        print(f"✅ 텍스트를 {len(splits)}개의 청크로 분할했습니다.")
        
        # 임베딩 생성
        embeddings = OllamaEmbeddings(model=CONFIG['ollama']['embedding_model'])
        
        # 벡터 저장소 생성
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=vectorstore_path,
            collection_name=collection_name
        )
        vectorstore.persist()
        
        # 캐시에 저장
        _vectorstore_cache[file_hash] = vectorstore
        print(f"✅ 벡터 저장소 생성 및 캐싱 완료. (폴더: {collection_name})\n")
        
        return vectorstore
        
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ 파일을 찾을 수 없습니다: {file_path}")
    except Exception as e:
        raise Exception(f"❌ 벡터 저장소 생성 중 오류 발생 ({Path(file_path).name}): {str(e)}")

def get_multiple_vectorstores(file_paths: List[str]) -> List[Chroma]:
    """여러 PDF 파일의 벡터 저장소를 가져옵니다."""
    vectorstores = []
    for file_path in file_paths:
        try:
            vs = get_or_create_vectorstore(file_path)
            vectorstores.append(vs)
        except Exception as e:
            print(f"⚠️ 파일 처리 중 오류 발생: {Path(file_path).name} - {str(e)}")
            continue
    return vectorstores

# 문서 포맷팅 (출처 정보 포함)
def format_docs(docs: List[Document], source_info: dict = None) -> str:
    """검색된 문서들을 포맷팅합니다."""
    if not docs:
        return ""
    
    formatted_parts = []
    for i, doc in enumerate(docs):
        # 출처 정보가 있으면 추가
        source = ""
        if source_info and i < len(source_info):
            source = f"[출처: {source_info[i]}] "
        elif hasattr(doc, 'metadata') and doc.metadata.get('source'):
            source = f"[출처: {Path(doc.metadata['source']).name}] "
        
        formatted_parts.append(f"{source}{doc.page_content}")
    
    return "\n\n---\n\n".join(formatted_parts)

# RAG 체인 동작 (다중 PDF 지원)
def rag_chain(files: List, question: str) -> str:
    """RAG 파이프라인을 실행합니다. 여러 PDF 파일을 지원합니다."""
    if not files or len(files) == 0:
        return "❌ PDF 파일을 하나 이상 업로드해주세요."
    
    if not question or not question.strip():
        return "❌ 질문을 입력해주세요."
    
    try:
        # 파일 경로 리스트 생성
        file_paths = [file.name for file in files if file]
        
        if not file_paths:
            return "❌ 유효한 PDF 파일이 없습니다."
        
        print(f"\n📚 총 {len(file_paths)}개의 PDF 파일을 처리합니다:")
        for i, path in enumerate(file_paths, 1):
            print(f"   {i}. {Path(path).name}")
        
        # 여러 PDF의 벡터 저장소 가져오기
        vectorstores = get_multiple_vectorstores(file_paths)
        
        if not vectorstores:
            return "❌ 모든 PDF 파일 처리에 실패했습니다. 파일이 올바른지 확인해주세요."
        
        print(f"\n🔍 {len(vectorstores)}개의 벡터 저장소에서 검색을 시작합니다...")
        
        # 모든 벡터 저장소에서 검색
        all_retrieved_docs = []
        source_info = []
        
        for i, vectorstore in enumerate(vectorstores):
            retriever = vectorstore.as_retriever()
            retrieved_docs = retriever.invoke(question)
            
            if retrieved_docs:
                all_retrieved_docs.extend(retrieved_docs)
                # 각 문서에 출처 정보 추가
                file_name = Path(file_paths[i]).name
                source_info.extend([file_name] * len(retrieved_docs))
                print(f"   ✅ {file_name}: {len(retrieved_docs)}개 문서 검색됨")
        
        if not all_retrieved_docs:
            return "❌ 관련 문서를 찾을 수 없습니다. 질문을 더 구체적으로 작성해 보거나 다른 PDF를 사용해 보세요."
        
        print(f"✅ 총 {len(all_retrieved_docs)}개의 관련 문서를 찾았습니다.\n")
        
        # 컨텍스트 구성 (중복 제거 및 상위 결과만 선택)
        # 유사도 점수로 정렬 (있는 경우)
        unique_docs = []
        seen_content = set()
        for doc in all_retrieved_docs:
            content_hash = hash(doc.page_content[:100])  # 첫 100자로 중복 체크
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        # 상위 10개만 선택 (너무 많은 컨텍스트 방지)
        selected_docs = unique_docs[:10]
        context = format_docs(selected_docs, source_info[:len(selected_docs)])
        
        print(f"📝 최종 컨텍스트 미리보기:\n{context[:500]}...\n")
        
        # 프롬프트 구성
        pdf_names = ", ".join([Path(p).name for p in file_paths])
        prompt = f"""다음은 여러 PDF 문서에서 검색된 내용입니다.

PDF 파일들: {pdf_names}

Question: {question}

Context from PDFs:
{context}

위의 컨텍스트를 바탕으로 질문에 답변해주세요. 여러 PDF에서 정보를 찾았다면, 각 출처를 명시해주세요."""
        
        # LLM 호출
        print(f"🤖 LLM 모델 '{CONFIG['ollama']['llm_model']}'에 질문을 전송합니다...")
        response = ollama.chat(
            model=CONFIG['ollama']['llm_model'],
            messages=[
                {
                    "role": "system",
                    "content": CONFIG['system_prompt']
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        result = response['message']['content']
        print(f"✅ 답변 생성 완료.\n")
        return result
        
    except FileNotFoundError as e:
        return f"❌ 파일 오류: {str(e)}"
    except ValueError as e:
        return f"❌ 값 오류: {str(e)}"
    except ConnectionError as e:
        return f"❌ 연결 오류: Ollama 서버에 연결할 수 없습니다. Ollama가 실행 중인지 확인해주세요.\n상세: {str(e)}"
    except Exception as e:
        return f"❌ 예상치 못한 오류 발생: {str(e)}\n\n문제가 지속되면 개발자에게 문의해주세요."

# Gradio 인터페이스 (output 창만 하늘색 배경 적용)
# CSS를 사용하여 output Textbox만 하늘색 배경 적용
custom_css = """
/* Output Textbox만 하늘색 배경 적용 */
#output-textbox textarea {
    background: linear-gradient(135deg, #E0F7FA 0%, #B3E5FC 50%, #81D4FA 100%) !important;
    border-color: #4FC3F7 !important;
    color: #000000 !important;
}

#output-textbox .wrap {
    background: linear-gradient(135deg, #E0F7FA 0%, #B3E5FC 50%, #81D4FA 100%) !important;
}

/* Output 영역의 블록 배경도 하늘색으로 */
#output-textbox {
    background: linear-gradient(135deg, #E0F7FA 0%, #B3E5FC 50%, #81D4FA 100%) !important;
    border-radius: 8px;
    padding: 10px;
}

/* Output 블록의 전체 컨테이너 */
.gradio-container div[id*="output"] {
    background: linear-gradient(135deg, #E0F7FA 0%, #B3E5FC 50%, #81D4FA 100%) !important;
}
"""

# Blocks를 사용하여 CSS 적용 (Gradio 6.0에서는 css를 launch()로 이동)
with gr.Blocks(title=CONFIG['gradio']['title']) as iface:
    gr.Markdown(f"## {CONFIG['gradio']['title']}")
    gr.Markdown(CONFIG['gradio']['description'])
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(
                label="PDF 파일 업로드 (여러 개 선택 가능)",
                type="filepath",
                file_count="multiple",
                file_types=[".pdf"]
            )
            question_input = gr.Textbox(
                label="질문을 입력하세요",
                placeholder="예: 이 문서들의 주요 내용은 무엇인가요? 또는 여러 PDF에서 공통으로 다루는 주제는 무엇인가요?",
                lines=3
            )
            submit_btn = gr.Button("질문하기", variant="primary")
    
    with gr.Row():
        output_textbox = gr.Textbox(
            label="답변",
            lines=CONFIG.get('gradio', {}).get('output', {}).get('lines', 20),
            max_lines=CONFIG.get('gradio', {}).get('output', {}).get('max_lines', 50),
            autofocus=False,
            elem_id="output-textbox"
        )
    
    # 이벤트 연결
    submit_btn.click(
        fn=rag_chain,
        inputs=[file_input, question_input],
        outputs=output_textbox
    )
    
    question_input.submit(
        fn=rag_chain,
        inputs=[file_input, question_input],
        outputs=output_textbox
    )

if __name__ == "__main__":
    print("🚀 다중 PDF RAG 시스템을 시작합니다...")
    print(f"📋 설정 정보:")
    print(f"   - LLM 모델: {CONFIG['ollama']['llm_model']}")
    print(f"   - 임베딩 모델: {CONFIG['ollama']['embedding_model']}")
    print(f"   - 청크 크기: {CONFIG['text_splitter']['chunk_size']}")
    print(f"   - 청크 겹침: {CONFIG['text_splitter']['chunk_overlap']}")
    print(f"   - PDF 폴더: {CONFIG.get('pdfs_directory', 'pdfs')}")
    print(f"   - 벡터 저장소: {CONFIG['vectorstore']['cache_directory']}\n")
    # Gradio 6.0에서는 css를 launch() 메서드로 전달
    iface.launch(css=custom_css)
