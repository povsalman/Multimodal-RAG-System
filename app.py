import streamlit as st
import os
from PIL import Image
from rag_engine import query_rag_system
import pytesseract
from sentence_transformers import SentenceTransformer

# Tesseract Configuration
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Page Config
st.set_page_config(page_title="Multimodal RAG - Assignment 3", layout="wide")

# CSS for Chat Interface
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

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=50)
    st.title("Assignment 3: GenAI")
    st.markdown("---")
    st.header("Settings")
    
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
st.markdown("Ask questions about your financial documents and charts, or upload an image query.")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize last_query tracker (fix for loop)
if "last_query" not in st.session_state:
    st.session_state.last_query = None

# Initialize uploaded_image tracker
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# Display Chat History
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="user-msg">👤 <b>You:</b> {message["content"]}</div>', unsafe_allow_html=True)
        # Display uploaded image if present in message
        if "image" in message and message["image"] is not None:
            st.image(message["image"], caption="Your uploaded image", width=200)
    else:
        st.markdown(f'<div class="bot-msg">🤖 <b>AI:</b> {message["content"]}</div>', unsafe_allow_html=True)
        if "sources" in message:
            with st.expander("📄 View Retrieved Sources"):
                for i, doc in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1}:** {doc.metadata['source']} (Page {doc.metadata.get('page', 'N/A')}) - Type: {doc.metadata['type']}")
                    st.text(doc.page_content[:200] + "...")
                    if doc.metadata['type'] in ['image', 'image_text'] and 'image_path' in doc.metadata and os.path.exists(doc.metadata['image_path']):
                        st.image(doc.metadata['image_path'], caption="Retrieved Visual", width=300)

# Input Area - Image uploader and chat input side by side
col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.chat_input("Type your question here or upload an image and click send...")

with col2:
    uploaded_image = st.file_uploader("📎", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

# Logic Handling
query_text = None
is_image = False
query_embed = None
extracted_text = ""

if user_input and user_input != st.session_state.last_query:
    # Store the image for display (before processing)
    image_to_display = None
    
    # Check if there's an uploaded image to process
    if uploaded_image:
        image = Image.open(uploaded_image)
        image_to_display = image.copy()  # Copy for display
        
        # Extract text and embeddings
        extracted_text = pytesseract.image_to_string(image)
        query_text = user_input if user_input.strip() else "Find related information to this image."
        if extracted_text.strip():
            query_text += f" [Image OCR: {extracted_text[:200]}...]"
        
        embed_model = SentenceTransformer('clip-ViT-B-32')
        query_embed = embed_model.encode(image).tolist()
        is_image = True
        
        # Add user message with image
        st.session_state.messages.append({
            "role": "user", 
            "content": f"[Image Query] {user_input if user_input.strip() else 'Analyze this image'}",
            "image": image_to_display
        })
    else:
        # Text-only query
        query_text = user_input
        st.session_state.messages.append({"role": "user", "content": query_text})
    
    st.session_state.last_query = user_input
    
    # Rerun to show user message first
    st.rerun()

# Process Query
if query_text:
    with st.spinner(f"Thinking using {selected_model} ({strategy})..."):
        try:
            response, sources = query_rag_system(
                query_text, 
                strategy=strategy, 
                k_retrieval=k_val, 
                model_option=selected_model,
                is_image=is_image,
                query_embed=query_embed if is_image else None,
                extracted_text=extracted_text if is_image else ""
            )
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "sources": sources
            })
            
            # Clear the query_text to prevent reprocessing
            query_text = None
            
            st.rerun()
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
elif len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
    # There's a user message without a response - need to process it
    last_message = st.session_state.messages[-1]
    
    # Reconstruct query parameters from last message
    if "image" in last_message and last_message["image"] is not None:
        image = last_message["image"]
        is_image = True
        
        with st.spinner("Analyzing your image query..."):
            extracted_text = pytesseract.image_to_string(image)
            query_text = last_message["content"]
            
            embed_model = SentenceTransformer('clip-ViT-B-32')
            query_embed = embed_model.encode(image).tolist()
    else:
        query_text = last_message["content"]
        is_image = False
        query_embed = None
        extracted_text = ""
    
    with st.spinner(f"Thinking using {selected_model} ({strategy})..."):
        try:
            response, sources = query_rag_system(
                query_text, 
                strategy=strategy, 
                k_retrieval=k_val, 
                model_option=selected_model,
                is_image=is_image,
                query_embed=query_embed if is_image else None,
                extracted_text=extracted_text if is_image else ""
            )
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "sources": sources
            })
            
            st.rerun()
            
        except Exception as e:
            st.error(f"An error occurred: {e}")