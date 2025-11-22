import streamlit as st
import os
from PIL import Image
from rag_engine import query_rag_system
import pytesseract

# Page Config
st.set_page_config(page_title="Multimodal RAG - Assignment 3", layout="wide")

# CSS for Chat Interface (Visuals)
st.markdown("""
<style>
    .user-msg {
        background-color: #e6f3ff;
        color: black; 
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        text-align: right;
        border: 1px solid #ccc;
    }
    .bot-msg {
        background-color: #f0f2f6;
        color: black;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 5px solid #ff4b4b;
        border-right: 1px solid #ccc;
        border-top: 1px solid #ccc;
        border-bottom: 1px solid #ccc;
    }
    .source-box {
        font-size: 0.8em;
        color: #333;
        border-top: 1px solid #ddd;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=50)
    st.title("Assignment 3: GenAI")
    st.markdown("---")
    st.header("Settings")
    
    # --- Model Selection ---
    selected_model = st.selectbox(
        "Select LLM Model",
        ["Phi-3 (Local)", "Mistral (Local)", "Gemini (API)", "Groq (API)"]
    )
    
    strategy = st.selectbox(
        "Prompting Strategy",
        ["Zero-shot", "Chain-of-Thought (CoT)", "Few-shot"]
    )
    k_val = st.slider("Chunks to Retrieve", min_value=1, max_value=5, value=3)
    
    st.info(f"Running with: {selected_model}")
    st.caption("Built with LangChain & ChromaDB")

# Main Header
st.title("🔍 Multimodal RAG Agent")
st.markdown("Ask questions about your financial documents and charts.")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="user-msg">👤 <b>You:</b> {message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">🤖 <b>AI:</b> {message["content"]}</div>', unsafe_allow_html=True)
        if "sources" in message:
            with st.expander("📄 View Retrieved Sources"):
                for i, doc in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1}:** {doc.metadata['source']} (Page {doc.metadata['page']})")
                    st.text(doc.page_content[:200] + "...")
                    if doc.metadata['type'] == 'image' and 'image_path' in doc.metadata:
                        if os.path.exists(doc.metadata['image_path']):
                            st.image(doc.metadata['image_path'], caption="Retrieved Chart", width=300)

# Input Area
col1, col2 = st.columns([4, 1])

with col1:
    user_input = st.chat_input("Type your question here...")

with col2:
    uploaded_image = st.file_uploader("Or upload an image query", type=["png", "jpg", "jpeg"])

# Logic Handling
query_text = None

if user_input:
    query_text = user_input
    st.session_state.messages.append({"role": "user", "content": query_text})

elif uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Your Image Query", width=200)
    
    with st.spinner("Analyzing your image query..."):
        extracted_text = pytesseract.image_to_string(image)
        if extracted_text.strip():
            query_text = f"Find info related to this data: {extracted_text}"
            st.session_state.messages.append({"role": "user", "content": f"[Uploaded Image Query] Extracted keywords: {extracted_text[:50]}..."})
        else:
            st.error("Could not extract text from image. Please try a clearer image or type text.")

# Process Query
if query_text:
    with st.spinner(f"Thinking using {selected_model} ({strategy})..."):
        try:
            response, sources = query_rag_system(
                query_text, 
                strategy=strategy, 
                k_retrieval=k_val, 
                model_option=selected_model
            )
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "sources": sources
            })
            
            st.rerun()
            
        except Exception as e:
            st.error(f"An error occurred: {e}")