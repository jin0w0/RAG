import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import trim_messages

load_dotenv("../data/.env")
api_key = os.getenv("OPENAI_API_KEY")

@st.cache_resource
def process_pdf():
    loader = PyPDFLoader("../data/2024_KB_부동산_보고서_최종.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

@st.cache_resource
def initialize_vector_store():
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    faiss_path = "./faiss_vectorstore" # 저장할 FAISS 경로 지정
    
    # 2. 이미 저장된 FAISS vectorstore가 존재할 경우 로드
    if os.path.exists(faiss_path):
        return FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    
    # 1. 존재하지 않는 경우 새로 만들고 현재 폴더에 저장
    chunks = process_pdf()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(faiss_path)
    return vectorstore

@st.cache_resource
def initialize_chain():
    vectorstore = initialize_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    template = """당신은 KB 부동산 보고서 전문가입니다. 다음 정보를 바탕으로 사용자의 질문에 답변해주세요.
    컨텍스트: {context}
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            ("placeholder", "{chat_history}"),
            ("human", "{question}")
        ])
    model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=api_key)
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    base_chain = (RunnablePassthrough.assign(
        context=lambda x: format_docs(retriever.invoke(x["question"])),
        chat_history=lambda x: trim_messages(x["chat_history"], max_tokens=4, token_counter=len, strategy="last")
    ) 
    | prompt 
    | model
    |StrOutputParser()
    )
    chat_history = ChatMessageHistory()

    return RunnableWithMessageHistory(
        base_chain, 
        lambda session_id: chat_history,
        input_messages_key="question",
        history_messages_key="chat_history",
        )

def main():
    st.set_page_config(page_title="KB 부동산 보고서 챗봇", page_icon="🏠")
    st.title("🏠KB 부동산 보고서 AI 어드바이저")
    st.caption("2024 KB 부동산 보고서 기반 질의응답 시스템")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("질문을 입력하세요..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        chain = initialize_chain()

        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                response = chain.invoke(
                    {"question": prompt},
                    {"configurable": {"session_id": "streamlit_session"}}
                )
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

from pyngrok import ngrok
if __name__ == "__main__":
    # 1. 현재 열려있는 ngrok 터널이 있는지 확인 (Streamlit 재실행 시 중복 생성 방지)
    tunnels = ngrok.get_tunnels()
    
    if not tunnels:
        # 2. 터널이 없으면 새로 생성 (8501 포트 연결)
        # 참고: 이전 단계에서 터미널에 토큰 인증을 마쳤다면 알아서 설정값을 불러옵니다.
        public_url = ngrok.connect(8501)
        print(f"🌍 앱 접속 URL (새로 생성됨): {public_url.public_url}")
    else:
        # 3. 이미 터널이 있다면 기존 URL 출력
        print(f"🌍 앱 접속 URL (기존 유지): {tunnels[0].public_url}")
    
    # Streamlit 메인 함수 실행
    main()