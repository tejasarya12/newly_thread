"""
Nodes for LangGraph (Legacy/Batch Mode)
Kept for compatibility with existing structure.
"""
from typing import TypedDict, List, Optional

class SupportState(TypedDict):
    user_input: str
    answer: str

def input_router(state: SupportState):
    return state

def intent_analyzer(state: SupportState):
    return state

def retriever_node(state: SupportState):
    return state

def response_generator(state: SupportState):
    return state

def accuracy_evaluator(state: SupportState):
    return state

def output_router(state: SupportState):
    return state

def check_accuracy_route(state: SupportState):
    return "output_router"



















"""
from typing import TypedDict, List, Optional
import os
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
import logging

# Import local model utilities
from utils.model_loader import get_llm, get_embeddings, generate_response
from config import VECTOR_DB, ACCURACY_THRESHOLD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupportState(TypedDict):
 
    user_input: str
    mode_input: str  # "text" or "voice"
    mode_output: str  # "text", "voice", or "email"
    intent: str
    retrieved_docs: List[dict]
    answer: str
    accuracy: bool
    email: str
    audio_file: Optional[str]
    confidence_score: float

def input_router(state: SupportState) -> SupportState:
    
    logger.info("🔄 Input Router Node")
    
    if state["mode_input"] == "voice":
        # Voice-to-text is handled on frontend via Web Speech API
        # If server-side transcription is needed, implement here
        logger.info("  Voice input detected - already transcribed on client")
    else:
        logger.info(f"  Text input: {state['user_input'][:50]}...")
    
    return state

def intent_analyzer(state: SupportState) -> SupportState:
    
    logger.info("🎯 Intent Analyzer Node")
    
    try:
        # Use local LLM to classify intent
        system_prompt =
        
        prompt = f"Query: {state['user_input']}\n\nIntent category:"
        
        response = generate_response(prompt, system_prompt)
        
        # Extract intent from response
        intent = response.strip().lower()
        if intent not in ["product_info", "troubleshooting", "feature_request", "complaint", "general"]:
            # Fallback to keyword-based detection
            query_lower = state["user_input"].lower()
            if any(word in query_lower for word in ["how", "what", "explain", "tell me", "describe"]):
                intent = "product_info"
            elif any(word in query_lower for word in ["problem", "issue", "error", "not working", "broken", "fail"]):
                intent = "troubleshooting"
            elif any(word in query_lower for word in ["feature", "add", "want", "need", "wish", "could you"]):
                intent = "feature_request"
            elif any(word in query_lower for word in ["complaint", "unhappy", "disappointed", "terrible"]):
                intent = "complaint"
            else:
                intent = "general"
        
        state["intent"] = intent
        logger.info(f"  Detected intent: {intent}")
        
    except Exception as e:
        logger.error(f"Intent analysis failed: {e}")
        state["intent"] = "general"
    
    return state

def retriever_node(state: SupportState) -> SupportState:
   
    logger.info("📚 Retriever Node")
    
    try:
        # Load local embeddings model (sentence-transformers)
        embeddings = get_embeddings()
        
        # Load ChromaDB vector store
        vectorstore = Chroma(
            persist_directory=VECTOR_DB['persist_directory'],
            embedding_function=embeddings,
            collection_name=VECTOR_DB['collection_name']
        )
        
        # Perform similarity search
        docs = vectorstore.similarity_search(
            state["user_input"], 
            k=3  # Retrieve top 3 most relevant documents
        )
        
        state["retrieved_docs"] = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            } 
            for doc in docs
        ]
        
        logger.info(f"  Retrieved {len(state['retrieved_docs'])} documents")
        if state['retrieved_docs']:
            logger.info(f"  Top source: {state['retrieved_docs'][0]['metadata'].get('source', 'unknown')}")
        
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        state["retrieved_docs"] = []
    
    return state

def response_generator(state: SupportState) -> SupportState:
    
    logger.info("✍️ Response Generator Node")
    
    try:
        # Prepare context from retrieved documents
        if state["retrieved_docs"]:
            context = "\n\n".join([
                f"Document {i+1} (from {doc['metadata'].get('source', 'unknown')}):\n{doc['content']}" 
                for i, doc in enumerate(state["retrieved_docs"])
            ])
        else:
            context = "No specific documentation found."
        
        # Create prompt with context for DeepSeek
        system_prompt = 
        
        
        
        # Generate response using local DeepSeek LLM
        answer = generate_response(prompt, system_prompt)
        state["answer"] = answer.strip()
        
        logger.info(f"  Generated answer: {state['answer'][:100]}...")
        
    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        state["answer"] = "I apologize, but I'm having trouble generating a response right now. Please try rephrasing your question or contact support directly."
    
    return state

def accuracy_evaluator(state: SupportState) -> SupportState:
   
    logger.info("🎯 Accuracy Evaluator Node")
    
    try:
        # Use local L
        
        response = generate_response(prompt, system_prompt)
        
        # Extract confidence score
        try:
            confidence = float(response.strip().split()[0])  # Get first number
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
        except:
            # Fallback: simple heuristic based on answer quality
            if len(state["retrieved_docs"]) > 0 and len(state["answer"]) > 50:
                confidence = 0.75
            elif len(state["retrieved_docs"]) > 0:
                confidence = 0.60
            else:
                confidence = 0.40
        
        state["confidence_score"] = confidence
        state["accuracy"] = confidence >= ACCURACY_THRESHOLD
        
        logger.info(f"  Accuracy: {state['accuracy']}, Confidence: {confidence:.2f}")
        
    except Exception as e:
        logger.error(f"Accuracy evaluation failed: {e}")
        # Safe fallback
        state["accuracy"] = len(state["retrieved_docs"]) > 0
        state["confidence_score"] = 0.70 if state["accuracy"] else 0.40
    
    return state

def output_router(state: SupportState) -> SupportState:
   
    logger.info("📤 Output Router Node")
    
    if state["mode_output"] == "voice":
        try:
            # Use pyttsx3 for offline text-to-speech
            import pyttsx3
            from config import TTS_CONFIG
            
            engine = pyttsx3.init()
            engine.setProperty('rate', TTS_CONFIG['rate'])
            engine.setProperty('volume', TTS_CONFIG['volume'])
            
            # Generate unique filename
            filename = f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
            filepath = f"static/audio/{filename}"
            
            # Save audio file
            engine.save_to_file(state["answer"], filepath)
            engine.runAndWait()
            
            state["audio_file"] = f"/audio/{filename}"
            logger.info(f"  TTS audio saved: {filename}")
            
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            state["audio_file"] = None
            logger.warning("  Falling back to text-only output")
        
    elif state["mode_output"] == "email":
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            from config import EMAIL_CONFIG
            
            if not EMAIL_CONFIG['sender_email'] or not EMAIL_CONFIG['sender_password']:
                logger.warning("Email credentials not configured in .env")
                return state
            
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = EMAIL_CONFIG['sender_email']
            msg['To'] = state["email"]
            msg['Subject'] = "Your Product Support Query Response"
            
            body = 
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
            server.starttls()
            server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"  Email sent successfully to: {state['email']}")
            
        except Exception as e:
            logger.error(f"Email sending failed: {e}")
            logger.warning("  Please check EMAIL_CONFIG in config.py and .env file")
    
    else:
        logger.info("  Text output (default)")
    
    return state

# Decision function for accuracy evaluator
def check_accuracy_route(state: SupportState) -> str:
   
    if state["accuracy"] and state["confidence_score"] > ACCURACY_THRESHOLD:
        logger.info("  ✅ Accuracy acceptable - proceeding to output")
        return "output_router"
    else:
        # Check if we've already retried (prevent infinite loops)
        retry_count = state.get("retry_count", 0)
        if retry_count < 1:  # Allow one retry
            logger.warning(f"  ⚠️ Low accuracy ({state['confidence_score']:.2f}) - retriggering retrieval")
            state["retry_count"] = retry_count + 1
            return "retriever_node"
        else:
            logger.warning("  ⚠️ Max retries reached - proceeding with current answer")
            return "output_router"