import os
import pdfplumber
import pytesseract
from PIL import Image
from langchain_core.documents import Document
from uuid import uuid4
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
import logging

# --- CONFIGURATION ---
DATA_FOLDER = "Data"
DB_PATH = "vector_db"
IMAGE_OUTPUT_DIR = "extracted_images"
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

logging.getLogger("pdfminer").setLevel(logging.ERROR)

def table_to_markdown(table):
    if not table:
        return ""
    headers = table[0]
    rows = table[1:]
    md = "| " + " | ".join(str(cell) if cell is not None else "" for cell in headers) + " |\n"
    md += "| " + " --- |" * len(headers) + "\n"
    for row in rows:
        md += "| " + " | ".join(str(cell) if cell is not None else "" for cell in row) + " |\n"
    return md

def extract_text_and_images(pdf_path):
    print(f"Processing: {pdf_path}")
    chunks = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # 1. Extract Text
            text = page.extract_text()
            if text and text.strip():
                chunks.append(Document(
                    page_content=text,
                    metadata={
                        "source": os.path.basename(pdf_path),
                        "page": page_num + 1,
                        "type": "text",
                        "image_path": "N/A"
                    }
                ))
            
            # 2. Extract Tables
            tables = page.extract_tables()
            for table_idx, table in enumerate(tables):
                table_str = table_to_markdown(table)
                if table_str.strip():
                    chunks.append(Document(
                        page_content=f"[TABLE {table_idx+1} on Page {page_num+1}]\n{table_str}",
                        metadata={
                            "source": os.path.basename(pdf_path),
                            "page": page_num + 1,
                            "type": "table",
                            "image_path": "N/A"
                        }
                    ))
            
            # 3. Full Page Visual Extraction (to capture charts/flowcharts not in .images)
            try:
                page_img = page.to_image(resolution=200).original
                ocr_text = pytesseract.image_to_string(page_img)
                if ocr_text.strip():
                    image_filename = f"{os.path.basename(pdf_path)}_page{page_num+1}_{uuid4().hex[:8]}.png"
                    image_path = os.path.join(IMAGE_OUTPUT_DIR, image_filename)
                    page_img.save(image_path)
                    # Add for image collection
                    chunks.append(Document(
                        page_content=ocr_text,  # Still store OCR for LLM if retrieved
                        metadata={
                            "source": os.path.basename(pdf_path),
                            "page": page_num + 1,
                            "type": "image",
                            "image_path": image_path
                        }
                    ))
                    # Add separate for text collection (OCR as text chunk)
                    chunks.append(Document(
                        page_content=f"[PAGE VISUAL CONTENT - Page {page_num+1}]\nOCR Extracted: {ocr_text}",
                        metadata={
                            "source": os.path.basename(pdf_path),
                            "page": page_num + 1,
                            "type": "image_text",
                            "image_path": image_path
                        }
                    ))
            except Exception as e:
                print(f"Error processing full page image on page {page_num+1}: {e}")
            
            # 4. Individual Images (with filters to skip small/noisy ones)
            # Skip for "3. FYP-Handbook-2023.pdf" to avoid extracting grey header/footer designs as split images
            if not pdf_path.endswith("3. FYP-Handbook-2023.pdf"):
                for img_obj in page.images:
                    try:
                        x0, top, x1, bottom = img_obj['x0'], img_obj['top'], img_obj['x1'], img_obj['bottom']
                        cropped_image = page.crop((x0, top, x1, bottom)).to_image(resolution=200)
                        img = cropped_image.original
                        width, height = img.size
                        if width < 200 or height < 200:
                            continue
                        ocr_text = pytesseract.image_to_string(img)
                        if len(ocr_text.strip()) < 20:
                            continue
                        image_filename = f"{os.path.basename(pdf_path)}_p{page_num+1}_img{uuid4().hex[:8]}.png"
                        image_path = os.path.join(IMAGE_OUTPUT_DIR, image_filename)
                        img.save(image_path)
                        # Image chunk for image collection
                        chunks.append(Document(
                            page_content=ocr_text,
                            metadata={
                                "source": os.path.basename(pdf_path),
                                "page": page_num + 1,
                                "type": "image",
                                "image_path": image_path
                            }
                        ))
                        # OCR text chunk for text collection
                        chunks.append(Document(
                            page_content=f"[IMAGE on Page {page_num+1}]\nOCR Extracted: {ocr_text}",
                            metadata={
                                "source": os.path.basename(pdf_path),
                                "page": page_num + 1,
                                "type": "image_text",
                                "image_path": image_path
                            }
                        ))
                    except Exception as e:
                        pass

    return chunks

def create_vector_db():
    """
    Ingests and embeds chunks into ChromaDB.
    """
    all_docs = []
    
    pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in Data folder.")
        return

    for pdf_file in pdf_files:
        path = os.path.join(DATA_FOLDER, pdf_file)
        pdf_docs = extract_text_and_images(path)
        all_docs.extend(pdf_docs)
        
    print(f"Total chunks created: {len(all_docs)}")

    # Embedders
    text_embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vision_embed_model = SentenceTransformer('clip-ViT-B-32')

    # Chroma client
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Clear and create text_collection
    if "text_collection" in [c.name for c in client.list_collections()]:
        client.delete_collection("text_collection")
    text_collection = client.create_collection("text_collection")
    
    # Clear and create image_collection
    if "image_collection" in [c.name for c in client.list_collections()]:
        client.delete_collection("image_collection")
    image_collection = client.create_collection("image_collection")
    
    text_ids = []
    text_embeddings = []
    text_metadatas = []
    text_documents = []
    
    image_ids = []
    image_embeddings = []
    image_metadatas = []
    image_documents = []
    
    for doc in all_docs:
        id_str = uuid4().hex
        if doc.metadata['type'] in ['text', 'table', 'image_text']:
            emb = text_embed_model.embed_query(doc.page_content)
            text_ids.append(id_str)
            text_embeddings.append(emb)
            text_metadatas.append(doc.metadata)
            text_documents.append(doc.page_content)
        if doc.metadata['type'] == 'image':
            image = Image.open(doc.metadata['image_path'])
            emb = vision_embed_model.encode(image).tolist()
            image_ids.append(id_str)
            image_embeddings.append(emb)
            image_metadatas.append(doc.metadata)
            image_documents.append(doc.page_content)  # OCR for reference
    
    if text_ids:
        text_collection.add(
            ids=text_ids,
            embeddings=text_embeddings,
            metadatas=text_metadatas,
            documents=text_documents
        )
    
    if image_ids:
        image_collection.add(
            ids=image_ids,
            embeddings=image_embeddings,
            metadatas=image_metadatas,
            documents=image_documents
        )
    
    print("Vector Databases Created and Saved!")

if __name__ == "__main__":
    create_vector_db()