# WhyHireYou - RAG-based Candidate Evaluation System

## Overview

WhyHireYou는 RAG (Retrieval-Augmented Generation) 방식을 활용하여 헤드헌터들이 후보자의 PDF 이력서를 기반으로 "Why should we hire you?"와 같은 질문에 대한 답변을 생성할 수 있도록 돕는 시스템입니다. 이 프로젝트는 LangChain, Chroma Vector Database, 및 OpenAI를 사용하여 PDF 문서를 처리하고, 벡터화된 데이터를 검색하여 자연어 응답을 생성합니다.

## Features

- **PDF Parsing**: 후보자의 PDF 이력서를 로드하고 텍스트 데이터를 추출합니다.
- **Text Splitting**: 텍스트를 효율적으로 처리하기 위해 적절한 크기로 분할합니다.
- **Vector Search**: Chroma를 사용하여 벡터화된 데이터를 저장하고 검색합니다.
- **RAG Pipeline**: 검색된 데이터를 기반으로 OpenAI의 언어 모델을 사용하여 자연어 응답을 생성합니다.
- **Customizable Prompts**: 헤드헌터의 요구에 맞게 프롬프트를 쉽게 수정할 수 있습니다.

## Use Case

헤드헌터는 다음과 같은 방식으로 이 시스템을 사용할 수 있습니다:

1. 후보자의 PDF 이력서를 업로드합니다.
2. "Why should we hire you?"와 같은 질문을 입력합니다.
3. 시스템은 PDF 데이터를 분석하고, 관련 정보를 검색하여 자연어로 답변을 생성합니다.

## Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/whyhireyou.git
cd whyhireyou
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Set Up Environment Variables

Create a `.env` file in the root directory and add your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Load Candidate Resumes

Place the candidate's PDF resumes in the `data/resumes/` directory.

### Run the Notebook

Open `app.ipynb` in Jupyter Notebook or VS Code and execute the cells step-by-step.

### Ask Questions

Use the RetrievalQA chain to ask questions like:
- "Why should we hire you?"
- "What are your key achievements?"
- "Describe your leadership experience."

### Get Responses

The system will generate a detailed, context-aware response based on the candidate's resume.

## Code Explanation

### 1. Load PDF Resumes

The PyPDFLoader is used to extract text from PDF files:

```python
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/resumes/candidate_resume.pdf")
documents = loader.load()
```

### 2. Split Text into Chunks

The RecursiveCharacterTextSplitter ensures the text is split into manageable chunks for vectorization:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)
```

### 3. Create Vector Store

The Chroma vector database is used to store and retrieve vectorized chunks:

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./data/chroma_db"
)
```

### 4. Build Retrieval-Augmented QA Chain

The RetrievalQA chain retrieves relevant chunks and generates answers using OpenAI's language model:

```python
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

response = qa_chain.run("Why should we hire you?")
print(response)
```

## Customization

**Prompt Template**: Modify the prompt to suit your specific needs:

```python
from langchain.prompts import PromptTemplate

custom_prompt_template = """
You are an AI assistant for headhunters.
Use the following pieces of context to answer the question.
If you don't know the answer, just say that you don't know.

Context: {context}
Question: {question}

Answer:
"""

PROMPT = PromptTemplate(
    template=custom_prompt_template, 
    input_variables=["context", "question"]
)
```

**Chunk Size**: Adjust the `chunk_size` and `chunk_overlap` in the RecursiveCharacterTextSplitter to optimize performance for larger resumes.

## Dependencies

- Python 3.8+
- Required Python Libraries:
  - langchain
  - chromadb
  - openai
  - python-dotenv
  - PyPDF2

Install all dependencies using:

```bash
pip install langchain chromadb openai python-dotenv PyPDF2
```

## Future Improvements

- Add support for multilingual resumes.
- Integrate additional LLMs (e.g., Llama2) for cost-effective local inference.
- Enhance the UI for non-technical users.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- LangChain for the framework.
- OpenAI for the language model.
- Chroma for vector database support.

