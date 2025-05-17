from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

import os
from langchain.chat_models import ChatOpenAI

pdf = "data/Stream-Processing-with-Apache-Flink.pdf"
loader = PyPDFLoader(pdf)
documents = loader.load()
# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
texts = text_splitter.split_documents(documents)
print(texts)

# Load embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# Create vector store
vector_store = FAISS.from_documents(texts, embeddings)
vector_store.save_local("faiss_index")  # Save for reuse

model_version = "deepseek-ai/deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_version)
model = AutoModelForCausalLM.from_pretrained(model_version)
# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


# Define the retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 chunks
# Create a Hugging Face pipeline for text generation
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.7
)
# Wrap the pipeline for LangChain compatibility
llm = HuggingFacePipeline(pipeline=pipe)
# Define the Prompt Template
template = """
Use the following context to answer the question. If unsure, say "I don't know."
Context:
{context}
Question: {question}
Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])
# Define the RAG Chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)
query = "What is Flink?"
result = rag_chain({"query": query})
# Extract the generated answer
answer = result["result"].split("Answer:")[1].strip()
print(answer)