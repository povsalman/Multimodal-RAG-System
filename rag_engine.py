from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# --- CONFIG ---
DB_PATH = "vector_db"

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GOOGLE_API_KEY or not GROQ_API_KEY:
    raise ValueError("API keys missing in .env")

def load_dbs():
    # Text DB (with text embedder)
    text_embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    text_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=text_embedding_function,
        collection_name="text_collection"
    )
    
    # Image DB (dummy embed for init; we use by_vector)
    image_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=None,  # No need for query embed
        collection_name="image_collection"
    )
    return text_db, image_db

def get_llm(model_option):
    if model_option == "Gemini (API)":
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3
        )
    elif model_option == "Groq (API)":
        # EXACT CONFIGURATION FROM YOUR STARTER CODE
        return ChatGroq(
            model="openai/gpt-oss-20b",  # Exact model requested
            api_key=GROQ_API_KEY,
            temperature=1,               # As per starter code
            max_tokens=8192,             # Mapped from max_completion_tokens
            top_p=1,
            reasoning_effort="medium"
        )
    elif model_option == "Mistral (Local)":
        return ChatOllama(model="mistral", temperature=0.3)
    elif model_option == "Phi-3 (Local)":
        return ChatOllama(model="phi3", temperature=0.3)
    else:
        return ChatOllama(model="phi3", temperature=0.3)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_prompt_template(strategy="Zero-shot"):
    if strategy == "Chain-of-Thought (CoT)":
        template = """
        Use the following pieces of context to answer the question. 
        Think step-by-step. Break down the figures, processes, or logic before giving the final answer.
        Use data from charts/tables/images (marked as [TABLE], [IMAGE], [PAGE VISUAL]) carefully.
        
        Context: {context}
        
        Question: {question}
        
        Reasoning Step-by-Step:
        1. Identify key information...
        2. Perform necessary calculations or breakdowns...
        3. Formulate conclusion...
        
        Final Answer:
        """
    elif strategy == "Few-shot":
        template = """
        Use the context to answer. Examples:
        
        Example 1:
        Q: What is the revenue growth?
        A: Based on the table in section 2, revenue grew by 15% from $1M to $1.15M.
        
        Actual Context: {context}
        Actual Question: {question}
        
        Answer:
        """
    else:
        template = """
        You are an expert analyst. Use the provided context (including OCR from images/tables) to answer accurately.
        If the context doesn't have the information, say you don't know—do not guess or use external knowledge.
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """
    
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

def query_rag_system(query_text, strategy="Zero-shot", k_retrieval=5, model_option="Phi-3 (Local)", is_image=False, query_embed=None, extracted_text=""):
    text_db, image_db = load_dbs()
    llm = get_llm(model_option)
    prompt = get_prompt_template(strategy)
    
    retrieved_chunks = []
    
    if is_image:
        # Vision search on image_db
        if query_embed is not None:
            retrieved_image = image_db.similarity_search_by_vector(query_embed, k=k_retrieval)
            retrieved_chunks += retrieved_image
        # Text search on text_db using OCR'd query
        if extracted_text.strip():
            retriever_text = text_db.as_retriever(search_kwargs={"k": k_retrieval})
            retrieved_text = retriever_text.invoke(extracted_text)
            retrieved_chunks += retrieved_text
    else:
        # Text search on text_db
        retriever_text = text_db.as_retriever(search_kwargs={"k": k_retrieval})
        retrieved_text = retriever_text.invoke(query_text)
        retrieved_chunks += retrieved_text
    
    # Dedup by content
    unique_chunks = {doc.page_content: doc for doc in retrieved_chunks}.values()
    
    context = format_docs(unique_chunks)
    
    answer_chain = prompt | llm | StrOutputParser()
    
    response = answer_chain.invoke({"context": context, "question": query_text})
    
    return response, list(unique_chunks)