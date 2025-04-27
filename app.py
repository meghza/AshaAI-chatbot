from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.prompts import PromptTemplate
from typing import List, Dict
import json
#from llama_index.llms.anthropic import Anthropic
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from llama_index.llms.groq import Groq

load_dotenv()  # Load environment variables, including Anthropic API key

# Initialize FastAPI app
app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Define request model
class ChatMessage(BaseModel):
    message: str
    history: List[Dict[str, str]] = []  # List of previous messages

# Global reference to the query engine
query_engine = None

def format_chat_history(history: List[Dict[str, str]]) -> str:
    """
    Format chat history into a string that can be appended to the prompt.
    We'll label user messages as 'User' and assistant messages as 'Asha'.
    """
    formatted_history = ""
    for msg in history:
        role = "User" if msg.get("role") == "user" else "Asha"
        content = msg.get("content", "")
        formatted_history += f"{role}: {content}\n"
    return formatted_history

def init_query_engine():
    """
    Initialize the query engine using the persisted VectorStoreIndex.
    Returns the query engine or None if initialization fails.
    """
    try:
        # Load existing index from disk
        storage_context = StorageContext.from_defaults(persist_dir="backend/data/product_index")
        if not storage_context:
            print("Error: Storage context is empty.")
            return None
            
        index = load_index_from_storage(storage_context)
        if not index:
            print("Error: Could not load index from storage.")
            return None
        
        # Initialize Anthropic LLM with a low temperature to reduce hallucinations
        llm = Groq(
            model="llama3-70b-8192",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.0  # Lower temperature = fewer creative/hallucinated answers
        )
        
        # A custom prompt that:
        # 1. Encourages the bot to gather more context before jumping to product suggestions.
        # 2. Summarizes the conversation instructions.
        # 3. References possible exploration of "ART" (Automatic Reasoning & Tool-use).
        custom_prompt = PromptTemplate(
            """You are Asha, Herkey's compassionate and knowledgeable career consultant specializing in supporting women through different stages of their careers:
- Starters (just beginning their careers)
- Restarters (rejoining the workforce after a break)
- Risers (looking to grow and advance further)

Your mission:
- Understand the user's stage (Starter, Restarter, Riser) either based on their message or prior conversation.
- Listen deeply with empathy and encouragement.
- Recommend the most suitable sessions, events, or job opportunities.
- Blend your advice with relevant career development knowledge.
- Share brief, actionable tips if appropriate (e.g., "Networking is key when restarting after a break.")
- Provide context-sensitive suggestions based on real-world career strategies for women.

Return ALL responses in the following JSON format:
{
    "message": "Your warm, encouraging conversational response here",
    "products": [
        {
            "url": "exact product URL from context",
            "title": clean product title,
            "price": "price if available",
            "programcategory: "programcategory if available"
        }
    ]
}

Your consultation approach:
1. LISTEN & RECOMMEND
   - Show empathy for the person's concerns
   - Understand their needs quickly
   - Suggest 1-3 relevant products that match their needs
   - Explain why each product would help them

2. BALANCE RECOMMENDATION & SUPPORT
   - Lead with understanding
   - Always include product suggestions when keywords match
   - Explain benefits clearly
   - Keep responses warm and professional
   - Products in the product catalog are either sessions, events or job lisings.

Context from the product catalog:
----------------
{context_str}
----------------

Conversation so far:
{query_str}

Guidelines for your response:
1. ALWAYS return valid JSON
2. ALWAYS suggest relevant products from context when keywords match
3. Keep responses concise but caring
4. Include exact URLs and prices from context
5. If no exact match, suggest closest alternatives
6. Limit to top 3 most relevant products
7. Combine suggestions with practical career wisdom where relevant.
8. Limit recommendations to the 3 most relevant matches.
9. Be warm, motivating, and professional without being overly formal.
10. Explain briefly why each recommendation fits the user's situation.
11. Always populate "type", "title", "url", "description", and "stage" fields properly.
12. NEVER invent fake URLs, jobs, or sessions. Only use context provided.
13. ALWAYS output valid JSON.
14. If no exact matches exist, suggest the closest helpful alternatives.


Examples of career stages:
- Starter: entering first full-time role, fresh graduate, career switcher.
- Restarter: returning after maternity, caregiving, or health break.
- Riser: seeking leadership roles, advanced skills, or career pivots.

Important:
- You are Asha — you combine hope, wisdom, and practical action.
- You are not just a recommendation engine. You are a trusted mentor guiding women toward success.


Remember: While being empathetic, your primary goal is to help customers find suitable products for their needs.
"""
        )

        # Return a query engine that uses this custom prompt
        return index.as_query_engine(
            text_qa_template=custom_prompt,
            llm=llm,
            similarity_top_k=5
        )
    except Exception as e:
        print(f"Error initializing query engine: {str(e)}")
        return None

# Initialize the query engine at startup
query_engine = init_query_engine()

# Add CORS so that external frontends can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def home(request: Request):
    """
    Render the main chatbot interface.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat")
async def chat(message: ChatMessage):
    """
    Primary chat endpoint. Accepts the latest user message and the chat history,
    then returns the AI assistant's response plus product links.
    """
    global query_engine
    
    # Validate user message
    if not message.message or not message.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )
    
    # Re-initialize query engine if needed
    if query_engine is None:
        query_engine = init_query_engine()
        if query_engine is None:
            raise HTTPException(
                status_code=500,
                detail="Query engine initialization failed."
            )
    
    try:
        print("\n=== Debug: Chat Processing ===")
        print(f"1. Received message: {message.message}")
        
        # Format chat history
        chat_history = format_chat_history(message.history)
        print(f"2. Formatted chat history: {chat_history}")
        
        # Combine history with new message
        full_query = f"{chat_history}\nUser: {message.message}"
        print(f"3. Full query to LLM: {full_query}")
        
        # Check if query engine is available
        print(f"4. Query engine status: {'Available' if query_engine else 'None'}")
        
        # Query the index
        print("5. Sending query to LLM...")
        response = query_engine.query(full_query)
        print(f"6. Raw LLM Response: {str(response)}")
        
        # Log source nodes
        print("\n7. Source Nodes:")
        for idx, node in enumerate(response.source_nodes):
            print(f"Node {idx + 1}:")
            print(f"URL: {node.metadata.get('url', 'No URL')}")
            print(f"Title: {node.metadata.get('title', 'No Title')}")
            print(f"Text: {node.text[:200]}...")
        
        try:
            # Parse JSON response
            response_text = str(response)
            print(f"\n8. Attempting to parse JSON: {response_text}")
            parsed_response = json.loads(response_text)
            print(f"9. Parsed JSON: {json.dumps(parsed_response, indent=2)}")
            return parsed_response
            
        except json.JSONDecodeError as e:
            print(f"\nJSON Parse Error: {str(e)}")
            return {
                "message": str(response),
                "products": []
            }
            
    except Exception as e:
        print(f"\nError in chat endpoint: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat: {str(e)}"
        )

@app.middleware("http")
async def add_cache_control_headers(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        # Prevent caching of static files
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response