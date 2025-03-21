import streamlit as st
import os
from dotenv import load_dotenv
from langchain.vectorstores.elasticsearch import ElasticsearchStore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# 환경 변수 로드
load_dotenv()

# API 키 및 환경 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL")
ELASTICSEARCH_USERNAME = os.getenv("ELASTICSEARCH_USERNAME")
ELASTICSEARCH_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD")
INDEX_NAME = os.getenv("INDEX_NAME", "documents")

# 페이지 설정
st.set_page_config(page_title="RAG 챗봇", page_icon="🤖", layout="wide")

# 사이드바 설정
st.sidebar.title("RAG 챗봇 설정")
model_name = st.sidebar.selectbox("모델 선택", ["gpt-3.5-turbo", "gpt-4"])
# temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

# 메인 페이지 설정
st.title("RAG 기반 챗봇")
st.markdown("Elasticsearch와 LangChain을 활용한 RAG 기반 챗봇입니다.")



# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "conversation" not in st.session_state:
    # Elasticsearch 벡터 스토어 초기화
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    # Elasticsearch 연결 설정
    vector_store = ElasticsearchStore(
        es_url=ELASTICSEARCH_URL,
        index_name=INDEX_NAME,
        embedding=embeddings,
        es_user=ELASTICSEARCH_USERNAME,
        es_password=ELASTICSEARCH_PASSWORD
    )
    
    # 대화 메모리 초기화
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # LLM 모델 초기화
    llm = ChatOpenAI(
        model_name=model_name,
        # temperature=temperature,
        openai_api_key=OPENAI_API_KEY
    )
    
    # 대화 체인 생성
    st.session_state.conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        return_source_documents=True
    )

# 이전 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("무엇이든 물어보세요!"):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # AI 응답 생성
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("생각 중..."):
            response = st.session_state.conversation({"question": prompt})
            answer = response["answer"]
            
            # 출처 정보 추가
            if "source_documents" in response and response["source_documents"]:
                sources = set()
                for doc in response["source_documents"]:
                    if hasattr(doc, "metadata") and "source" in doc.metadata:
                        sources.add(doc.metadata["source"])
                
                if sources:
                    answer += "\n\n**출처:**\n"
                    for i, source in enumerate(sources, 1):
                        answer += f"{i}. {source}\n"
    
        message_placeholder.markdown(answer)
    
    # 응답 메시지 저장
    st.session_state.messages.append({"role": "assistant", "content": answer})
