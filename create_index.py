import csv
import requests
from bs4 import BeautifulSoup
from llama_index.core import Document
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

def load_products_from_csv(csv_path):
    """
    Load product data from a CSV file, validate, and prepare for indexing.
    """
    products = []

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=1):
            # Clean whitespace from all fields
            row = {k: (v.strip() if v else "") for k, v in row.items()}

            programdate = row.get("programdate", "")
            title = row.get("title", "")
            price = row.get("price", "")
            description = row.get("description", "")
            programcategory = row.get("programcategory", "")
            url = row.get("url", "")
            programtype = row.get("programtype", "")

            # Validation checks
            if not title:
                print(f"[WARN] Row {row_num} skipped: Missing title.")
                continue

            if not description:
                print(f"[WARN] Row {row_num}: Missing description. Proceeding anyway.")

            # Build combined text for document
            doc_text = (
                f"PRODUCT TITLE:\n{title}\n\n"
                f"PRICE:\n{price or 'Unknown'}\n\n"
                f"DESCRIPTION:\n{description}\n\n"
                f"PROGRAMCATEGORY:\n{programcategory}\n\n"
                f"PROGRAMTYPE:\n{programtype}\n"
            )

            metadata = {
                "url": url,
                "title": title,
                "price": price or "Unknown",
                "programtype": programtype or "Unknown",
                "type": "product"
            }

            products.append((doc_text, metadata))

    print(f"✅ Loaded {len(products)} valid products from CSV.")
    return products
    

def create_product_index():
    """
    Create a product index using data from the CSV file and save it for RAG applications.
    """
    # 1. Load and validate products
    products = load_products_from_csv("final data.csv")

    all_docs = []
       
    # 2. Create Document objects
    for doc_text, metadata in products:
        # Create one doc per product
        doc = Document(
            text=doc_text,
            metadata=metadata
        )
        all_docs.append(doc)
    
    print(f"Collected")
    
    # 3. Build the index
    #   - Use Anthropic with a low temperature
    llm = Anthropic(
        model="claude-3-sonnet-20240229",  # or claude-3 etc.
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.0
    )
    
    # Update the settings globally
    Settings.llm = llm
    
    # Create index directly without service_context
    index = VectorStoreIndex.from_documents(all_docs)
    
    # 4. Persist to disk
    storage_context = index.storage_context
    storage_context.persist(persist_dir="backend/data/product_index")

    print("Product index created and persisted successfully!")

if __name__ == "__main__":
    create_product_index()