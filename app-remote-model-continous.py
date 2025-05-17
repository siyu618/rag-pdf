import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv
load_dotenv()
# 设置 DeepSeek API

# 1. 加载 PDF 文档
pdf_path = "data/Stream-Processing-with-Apache-Flink.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# 2. 分割文本
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# 3. 嵌入向量生成 & 向量数据库
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = FAISS.from_documents(texts, embeddings)
vector_store.save_local("faiss_index")

# 4. 检索器
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 5. 使用 DeepSeek API 作为语言模型
llm = ChatOpenAI(
    model_name="deepseek-chat",
    temperature=0.7,
    max_tokens=512
)

# 6. Prompt 模板
template = """
Use the following context to answer the question. If unsure, say "I don't know."
Context:
{context}
Question: {question}
Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# 7. 构建 RetrievalQA Chain（RAG）
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# 8. 执行问答
# 构建对话记忆
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 构建持续对话 RAG 链
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    verbose=True
)

# 开始对话
print("💬 Chat with your PDF (type 'exit' to stop):\n")

while True:
    query = input("👤 You: ")
    if query.lower() in ["exit", "quit"]:
        break

    result = qa_chain({"question": query})
    answer = result["answer"]
    print("🤖 Bot:", answer)
