from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
import os
from dotenv import load_dotenv

# --- CONFIG ---
DB_PATH = "vector_db"

# --- LOAD ENVIRONMENT ---
load_dotenv()  # Loads variables from .env

# --- API KEYS FROM ENV ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment. Add it to .env")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment. Add it to .env")

def load_db():
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
    return db

def get_llm(model_option):
    """
    Returns the LLM object based on the user's selection.
    """
    if model_option == "Gemini (API)":
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
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
        Think step-by-step. Break down the financial figures or logic before giving the final answer.
        If the context contains data from charts (marked as [IMAGE DETECTED]), analyze that data carefully.
        
        Context: {context}
        
        Question: {question}
        
        Reasoning Step-by-Step:
        1. Identify key information...
        2. Perform necessary calculations...
        3. Formulate conclusion...
        
        Final Answer:
        """
    elif strategy == "Few-shot":
        template = """
        Use the context to answer. Here are examples of how to answer:
        
        Example 1:
        Q: What is the revenue growth?
        A: Based on the table in section 2, revenue grew by 15% from $1M to $1.15M.
        
        Actual Context: {context}
        Actual Question: {question}
        
        Answer:
        """
    else: # Zero-shot (Default)
        template = """
        You are an expert financial analyst. Use the provided context (including OCR text from images) to answer the user's question.
        If you don't know the answer, just say you don't know.
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """
    
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

def query_rag_system(query_text, strategy="Zero-shot", k_retrieval=3, model_option="Phi-3 (Local)"):
    # 1. Setup Resources
    db = load_db()
    retriever = db.as_retriever(search_kwargs={"k": k_retrieval})
    
    # Pass the selected model option to get the correct LLM
    llm = get_llm(model_option)
    
    prompt = get_prompt_template(strategy)
    
    # 2. Build LCEL Chain
    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )
    
    answer_chain = (
        RunnablePassthrough.assign(
            formatted_context=(lambda x: format_docs(x["context"]))
        )
        | (lambda x: {"context": x["formatted_context"], "question": x["question"]})
        | prompt
        | llm
        | StrOutputParser()
    )
    
    full_chain = setup_and_retrieval.assign(answer=answer_chain)
    
    result = full_chain.invoke(query_text)
    
    return result["answer"], result["context"]