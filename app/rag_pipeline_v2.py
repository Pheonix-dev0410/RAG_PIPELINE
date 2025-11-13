"""
Enhanced RAG Pipeline with Dual-Mode Operation
Mode 1: Video Link Retrieval
Mode 2: Transcript-based Q&A

Addresses all critical issues:
- Deprecated imports fixed
- Per-user session management
- Natural human-like responses
- User name personalization
- Transcript-based contextual answers
"""

from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from openai import OpenAI
import os
import logging
import re
import random
import pickle
from dotenv import load_dotenv

from app.config_v2 import (
    EMBEDDING_MODEL,
    FAISS_DIR,
    VIDEO_RETRIEVAL_CONFIG,
    TRANSCRIPT_RETRIEVAL_CONFIG,
    LLM_CONFIG_VIDEO_RETRIEVAL,
    LLM_CONFIG_TRANSCRIPT_QA,
    CONTENT_TYPE_VIDEO,
    CONTENT_TYPE_TRANSCRIPT,
    USE_HYBRID_SEARCH,
    HYBRID_DENSE_WEIGHT,
    HYBRID_SPARSE_WEIGHT,
)
from app.pdf_loader_v2 import load_and_split_pdf
from app.session_manager import UserSession, get_session_manager

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# ============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# ============================================================================

class VideoLinkResponse(BaseModel):
    """Structured output for video link retrieval or educational fallback"""
    video_link: Optional[str] = Field(None, description="The exact URL of the course/video (if available)")
    video_title: Optional[str] = Field(None, description="Title of the course (if available)")
    answer: Optional[str] = Field(None, description="Educational response when no video is available")
    
class TranscriptAnswer(BaseModel):
    """Structured output for transcript Q&A"""
    answer: str = Field(description="Natural, conversational answer based on transcript or course info")

# ============================================================================
# PROMPTS - Dual Mode
# ============================================================================

# Prompt for VIDEO LINK RETRIEVAL
VIDEO_RETRIEVAL_PROMPT = """You are Welida, a friendly and knowledgeable learning assistant powered by GPT-4o.

**YOUR JOB:**
1. If the user asks to learn something and a relevant video EXISTS in the "Available Videos" below, return that exact video link.
2. If NO relevant video exists, YOU become their teacher and provide a comprehensive educational lesson on the topic.

**CRITICAL RULES FOR VIDEO LINKS:**
- You MUST extract the video link EXACTLY as it appears in the "Available Videos" section below.
- DO NOT create, modify, or hallucinate video links - only use links that are explicitly provided.
- ONLY return a video_link if it exists in the list below.

**WHEN NO VIDEO EXISTS - YOU ARE THE TEACHER:**
Act as their personal tutor and provide a high-quality educational response:
1. Use their name warmly to create a personal connection
2. Provide a clear, engaging explanation of the topic (4-6 sentences)
3. Break down concepts in an easy-to-understand way
4. Use examples, analogies, or real-world applications where relevant
5. End by offering to dive deeper: "Would you like me to explain any specific aspect in more detail?"
6. Be enthusiastic and encouraging - make learning exciting!

**Example (when no video exists):**
"Hey Pranav! Let me teach you about quantum physics - it's absolutely fascinating! Quantum physics explores the behavior of matter and energy at the tiniest scales imaginable - we're talking about atoms, electrons, and photons. At this scale, particles don't follow the normal rules we see in everyday life. For example, an electron can actually be in multiple places at once until we observe it! Think of it like a coin spinning in the air - it's both heads and tails until it lands. This strange behavior is what makes quantum computers possible and why understanding quantum mechanics is revolutionizing technology. Would you like me to explain any specific aspect in more detail, like wave-particle duality or quantum entanglement?"

**Available Videos:**
{context}

**Chat History:**
{chat_history}

**User Question:** {question}

**Your Response (return video_link if it exists, otherwise teach the topic yourself):**"""

# Prompt for TRANSCRIPT Q&A (when transcripts exist)
TRANSCRIPT_QA_PROMPT = """You are Welida, a friendly and engaging video learning assistant. The user is asking about a video they recently watched.

**YOUR PERSONALITY:**
- Warm, encouraging, and conversational
- Use the user's name ({user_name}) naturally in responses
- Explain concepts clearly like a patient tutor
- Match the tone and teaching style from the video transcript

**RULES:**
1. Answer based ONLY on the video transcript provided in context.
2. Be natural and human - explain things in your own words based on the transcript.
3. NEVER say "not found", "I don't know", or "the transcript doesn't mention" - instead, provide the closest related information from the transcript.
4. If the question is about a concept mentioned in the video, explain it thoroughly.
5. Use examples from the transcript when possible.
6. Keep responses concise but informative (2-4 sentences usually).

**Video Transcript Context:**
{context}

**Recent Conversation:**
{chat_history}

**User's Question:** {question}

**Your Response (natural, warm, and helpful):**"""

# Prompt for FALLBACK Q&A (when NO transcripts available)
FALLBACK_QA_PROMPT = """You are Welida, a friendly and engaging course assistant.

The user is asking a follow-up question about a course they were interested in. Unfortunately, detailed transcripts are not available yet, but you have the course description.

**YOUR PERSONALITY:**
- Warm, encouraging, and conversational  
- Use the user's name ({user_name}) naturally in responses
- Be honest about limitations but helpful

**RULES:**
1. Answer based on the course description/metadata provided in context.
2. Be natural and conversational - never robotic.
3. If you can answer from the description, do so warmly.
4. If the description doesn't have enough detail, say something like:
   "Hey {user_name}, that's a great question! While I don't have the detailed video content right now, based on the course description, [provide what you can]. I'd recommend checking out the course to get the full explanation!"
5. NEVER say "transcript not available" or "not found" - keep it natural.
6. Keep responses concise but helpful (2-4 sentences).

**Course Information:**
{context}

**Recent Conversation:**
{chat_history}

**User's Question:** {question}

**Your Response (honest, warm, and helpful):**"""

# ============================================================================
# PROMPT TEMPLATES - Not needed anymore, using structured output directly
# ============================================================================
# Prompts are now built directly in query_with_structured_output() function

# ============================================================================
# PDF INGESTION
# ============================================================================

def ingest_pdf(pdf_path: str, backup: bool = True) -> bool:
    """
    Ingest a PDF file with improved error handling and backup.
    
    Args:
        pdf_path: Path to PDF file
        backup: Whether to backup existing index
        
    Returns:
        True if successful
    """
    try:
        logger.info(f"Starting PDF ingestion: {pdf_path}")
        
        # Validate
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        if os.path.getsize(pdf_path) == 0:
            raise ValueError(f"PDF is empty: {pdf_path}")
        
        # Backup existing index
        if backup and os.path.exists(os.path.join(FAISS_DIR, "index.faiss")):
            import shutil
            from datetime import datetime
            backup_dir = f"{FAISS_DIR}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copytree(FAISS_DIR, backup_dir)
            logger.info(f"Backed up existing index to: {backup_dir}")
        
        # Load and split with metadata
        docs = load_and_split_pdf(pdf_path)
        
        if not docs:
            raise ValueError("No documents extracted from PDF")
        
        logger.info(f"Extracted {len(docs)} document chunks")
        
        # Create FAISS index
        vectordb = FAISS.from_documents(
            documents=docs,
            embedding=EMBEDDING_MODEL
        )
        
        # Save FAISS index
        os.makedirs(FAISS_DIR, exist_ok=True)
        vectordb.save_local(FAISS_DIR)
        logger.info("FAISS index saved")
        
        # If hybrid search enabled, also save documents for BM25
        if USE_HYBRID_SEARCH:
            docs_path = os.path.join(FAISS_DIR, "documents.pkl")
            with open(docs_path, 'wb') as f:
                pickle.dump(docs, f)
            logger.info(f"Saved {len(docs)} documents for BM25 hybrid search")
        
        logger.info(f"✅ PDF ingestion complete: {len(docs)} chunks indexed")
        return True
        
    except Exception as e:
        logger.error(f"PDF ingestion failed: {e}", exc_info=True)
        raise

# ============================================================================
# RETRIEVER CREATION
# ============================================================================

def get_retriever(mode: str = "video", video_id: Optional[str] = None, allow_fallback: bool = True):
    """
    Get retriever configured for specific mode.
    Supports both pure FAISS and hybrid (FAISS + BM25) search.
    
    Args:
        mode: "video" or "transcript"
        video_id: Specific video ID to filter transcripts
        allow_fallback: If True and no transcripts found, falls back to video metadata
        
    Returns:
        Configured retriever (pure FAISS or hybrid)
    """
    try:
        # Load FAISS index
        if not os.path.exists(os.path.join(FAISS_DIR, "index.faiss")):
            raise FileNotFoundError(f"FAISS index not found in {FAISS_DIR}")
        
        vectordb = FAISS.load_local(
            FAISS_DIR,
            EMBEDDING_MODEL,
            allow_dangerous_deserialization=True
        )
        
        # Configure based on mode
        if mode == "video":
            config = VIDEO_RETRIEVAL_CONFIG.copy()
            content_filter = {"content_type": CONTENT_TYPE_VIDEO}
        else:
            config = TRANSCRIPT_RETRIEVAL_CONFIG.copy()
            content_filter = {"content_type": CONTENT_TYPE_TRANSCRIPT}
            if video_id:
                content_filter["video_id"] = video_id
        
        config["search_kwargs"]["filter"] = content_filter
        
        # Create dense retriever (FAISS)
        dense_retriever = vectordb.as_retriever(**config)
        
        # If hybrid search disabled, return pure FAISS
        if not USE_HYBRID_SEARCH:
            logger.debug(f"Created pure FAISS {mode} retriever")
            return dense_retriever
        
        # Hybrid search: Load documents and create BM25 retriever
        try:
            docs_path = os.path.join(FAISS_DIR, "documents.pkl")
            if not os.path.exists(docs_path):
                logger.warning(f"Documents not found for BM25. Falling back to pure FAISS.")
                return dense_retriever
            
            # Load documents
            with open(docs_path, 'rb') as f:
                all_docs = pickle.load(f)
            
            # Filter documents by content type
            filtered_docs = [
                doc for doc in all_docs
                if doc.metadata.get("content_type") == content_filter["content_type"]
            ]
            
            # Further filter by video_id if specified
            if video_id and "video_id" in content_filter:
                filtered_docs = [
                    doc for doc in filtered_docs
                    if doc.metadata.get("video_id") == video_id
                ]
            
            if not filtered_docs:
                logger.warning(f"No documents found for {mode} mode. Using pure FAISS.")
                return dense_retriever
            
            # Create BM25 retriever
            bm25_retriever = BM25Retriever.from_documents(filtered_docs)
            bm25_retriever.k = config["search_kwargs"].get("k", 5)
            
            # Create hybrid ensemble retriever
            ensemble_retriever = EnsembleRetriever(
                retrievers=[dense_retriever, bm25_retriever],
                weights=[HYBRID_DENSE_WEIGHT, HYBRID_SPARSE_WEIGHT]
            )
            
            logger.info(
                f"✅ Created HYBRID {mode} retriever: "
                f"{len(filtered_docs)} docs, "
                f"weights=[{HYBRID_DENSE_WEIGHT}, {HYBRID_SPARSE_WEIGHT}]"
            )
            
            return ensemble_retriever
            
        except Exception as e:
            logger.warning(f"Error creating hybrid retriever: {e}. Falling back to pure FAISS.")
            return dense_retriever
        
    except Exception as e:
        logger.error(f"Error creating retriever: {e}", exc_info=True)
        raise


def check_transcripts_available(video_id: Optional[str] = None) -> bool:
    """
    Check if transcripts are available in the vector store.
    
    Args:
        video_id: Optional specific video ID to check
        
    Returns:
        True if transcripts exist, False otherwise
    """
    try:
        if not os.path.exists(os.path.join(FAISS_DIR, "index.faiss")):
            logger.warning("FAISS index not found")
            return False
        
        vectordb = FAISS.load_local(
            FAISS_DIR,
            EMBEDDING_MODEL,
            allow_dangerous_deserialization=True
        )
        
        # Try to search for transcript content
        filter_dict = {"content_type": CONTENT_TYPE_TRANSCRIPT}
        if video_id:
            filter_dict["video_id"] = video_id
        
        logger.info(f"🔍 Checking transcripts with filter: {filter_dict}")
        
        # Perform a simple search to see if any transcripts exist
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 1, "filter": filter_dict}
        )
        
        # Try to retrieve documents
        test_docs = retriever.invoke("test query")
        
        has_transcripts = len(test_docs) > 0
        logger.info(f"Transcript availability check: {has_transcripts} (found {len(test_docs)} docs, video_id={video_id})")
        if test_docs:
            logger.debug(f"Sample transcript metadata: {test_docs[0].metadata}")
        
        return has_transcripts
        
    except Exception as e:
        logger.warning(f"Error checking transcript availability: {e}")
        return False

# ============================================================================
# OLD QA CHAIN CREATION - REMOVED (using structured output now)
# ============================================================================
# This function has been replaced by query_with_structured_output() below

# ============================================================================
# STRUCTURED OUTPUT QUERY (Using OpenAI directly)
# ============================================================================

def query_with_structured_output(
    question: str,
    context: str,
    chat_history: str,
    mode: str,
    user_name: str = "there"
) -> Dict[str, Any]:
    """
    Query using OpenAI's structured output API for guaranteed response format.
    
    Args:
        question: User's question
        context: Retrieved context from vector store
        chat_history: Conversation history
        mode: "video" or "transcript"
        user_name: User's name for personalization
        
    Returns:
        Structured response dict
    """
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Build system prompt based on mode
        if mode == "video":
            system_prompt = """You are Welida, a friendly and knowledgeable learning assistant powered by GPT-4o.

RULES:
1. If a relevant video EXISTS in the context, return the video_link (exact URL from context) and video_title.
2. If NO relevant video exists - YOU ARE THE TEACHER:
   - Set video_link and video_title to null
   - In the 'answer' field, provide a comprehensive educational lesson:
     * Use the user's name warmly to create personal connection
     * Provide clear, engaging explanation (4-6 sentences)
     * Break down concepts in easy-to-understand way
     * Use examples, analogies, or real-world applications
     * End with: "Would you like me to explain any specific aspect in more detail?"
     * Be enthusiastic and encouraging - make learning exciting!
3. NEVER make up URLs - only use URLs that actually exist in the context
4. When teaching (no video), provide HIGH-QUALITY educational content as if you're their personal tutor
"""
            response_format = VideoLinkResponse
        else:
            system_prompt = f"""You are Welida, a friendly and engaging learning assistant.
Answer the user's question naturally and conversationally using the provided context.

PERSONALITY:
- Warm, encouraging, and helpful
- Use the user's name ({user_name}) naturally
- Be honest about limitations but always helpful
- Never say "not found" or "I don't know" - provide closest related info

RULES:
1. Answer based on the context provided
2. Be natural and human-like
3. Keep responses concise but informative (2-4 sentences)
4. If context is limited, acknowledge it gracefully and provide what you can
"""
            response_format = TranscriptAnswer
        
        # Build user prompt with context
        user_prompt = f"""User's Name: {user_name}

Available Context:
{context}

Recent Conversation:
{chat_history}

User's Question: {question}

Provide your response:"""
        
        # Call OpenAI with structured output
        completion = client.beta.chat.completions.parse(
            model="gpt-4o",  # or "gpt-4o-2024-08-06" for latest
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=response_format,
            temperature=0.5 if mode == "video" else 0.7,  # Slightly higher for natural teaching tone
            max_tokens=1000 if mode == "video" else 800,  # High token limit for comprehensive teaching
        )
        
        # Parse the structured output
        parsed_response = completion.choices[0].message.parsed
        
        logger.info(f"✅ Structured output received: {parsed_response}")
        
        return parsed_response.model_dump()
        
    except Exception as e:
        logger.error(f"Error in structured output query: {e}", exc_info=True)
        raise

# ============================================================================
# RANDOMIZATION FOR DUPLICATE COURSES
# ============================================================================

def randomize_duplicate_courses(source_docs: List) -> List:
    """
    Randomize selection when multiple videos exist for the same course.
    
    Groups documents by course identifier (title + description) and randomly
    selects one document from each group. This ensures users get different
    videos each time they ask for the same course.
    
    Args:
        source_docs: List of retrieved documents
        
    Returns:
        List of documents with duplicates randomly selected
    """
    if not source_docs:
        return source_docs
    
    from collections import defaultdict
    
    # Group documents by course identifier
    course_groups = defaultdict(list)
    
    for doc in source_docs:
        # Create a unique identifier for the course based on title and description
        # Extract title and description from content
        content = doc.page_content.lower()
        
        # Try to extract title
        title_match = re.search(r'title\s*:\s*([^\n]+)', content, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else ""
        
        # Try to extract description
        desc_match = re.search(r'description\s*:\s*([^\n]+)', content, re.IGNORECASE)
        description = desc_match.group(1).strip() if desc_match else ""
        
        # Create identifier (use first 100 chars of title + description)
        course_id = f"{title[:100]}_{description[:100]}"
        
        # If no title/description found, treat each as unique (don't group)
        if not title and not description:
            course_id = f"unique_{id(doc)}"
        
        course_groups[course_id].append(doc)
    
    # Randomly select one document from each group
    randomized_docs = []
    for course_id, docs in course_groups.items():
        if len(docs) > 1:
            selected = random.choice(docs)
            logger.info(f"🎲 Randomized: {len(docs)} videos for course, selected one randomly")
            randomized_docs.append(selected)
        else:
            randomized_docs.append(docs[0])
    
    return randomized_docs


# ============================================================================
# MAIN QUERY INTERFACE
# ============================================================================

def query_rag_system(
    question: str,
    session_id: Optional[str] = None,
    user_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main interface for querying the RAG system.
    
    This function:
    1. Gets or creates user session
    2. Determines mode (video search or transcript Q&A)
    3. Checks transcript availability
    4. Creates appropriate QA chain (with fallback if needed)
    5. Processes the query
    6. Updates session state
    
    Args:
        question: User's question
        session_id: Optional session ID
        user_name: Optional user name for new sessions
        
    Returns:
        Dict with answer, video_link, session_id, mode
    """
    try:
        logger.info(f"Processing query: {question[:100]}...")
        
        # Get session manager
        manager = get_session_manager()
        
        # Get or create session
        session_id, session = manager.get_or_create_session(session_id, user_name)
        
        # Update user name if provided
        if user_name and not session.user_name:
            session.user_name = user_name
        
        # Determine mode: Is this a follow-up about the last video?
        mode = detect_query_mode(question, session)
        logger.info(f"Query mode: {mode}")
        
        # Check transcript availability if in transcript mode
        use_fallback = False
        if mode == "transcript":
            transcripts_available = check_transcripts_available(video_id=session.last_video_id)
            if not transcripts_available:
                logger.info("No transcripts available - using fallback mode")
                use_fallback = True
        
        # Get retriever and retrieve context
        retrieval_mode = mode if not use_fallback else "video"
        retriever = get_retriever(
            mode=retrieval_mode,
            video_id=session.last_video_id if mode == "transcript" and not use_fallback else None
        )
        
        # Retrieve relevant documents
        source_docs = retriever.invoke(question)
        
        # Apply randomization for video mode to ensure different videos for same course
        if mode == "video":
            source_docs = randomize_duplicate_courses(source_docs)
        
        # Build context string from retrieved documents
        context = "\n\n".join([doc.page_content for doc in source_docs])
        
        # Build chat history string
        chat_history = session.get_formatted_history(max_interactions=3)
        
        # Query with structured output
        structured_response = query_with_structured_output(
            question=question,
            context=context,
            chat_history=chat_history,
            mode=mode,
            user_name=session.user_name or "there"
        )
        
        # Process structured response
        result = process_structured_response(structured_response, source_docs, mode, session)
        
        # Update session
        session.add_interaction(
            question=question,
            answer=result["answer"] or result["video_link"],
            video_id=result.get("video_id")
        )
        
        manager.update_session(session_id, session)
        
        # Add session_id to result
        result["session_id"] = session_id
        result["mode"] = "transcript_fallback" if use_fallback else mode
        
        logger.info(f"✅ Query processed successfully: mode={mode}, fallback={use_fallback}, session={session_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def detect_query_mode(question: str, session: UserSession) -> str:
    """
    Detect whether this is a video search or transcript Q&A query.
    
    Heuristics:
    - If session has recent video AND question seems like follow-up → transcript mode
    - If question explicitly requests new video/topic → video mode
    - Default: video mode
    """
    question_lower = question.lower()
    
    # Context reference indicators (referring to previous video)
    context_references = [
        "this course", "that course", "this video", "that video",
        "the course", "the video", "it", "in this", "in that"
    ]
    
    # New topic indicators (strong signals for wanting a NEW course)
    new_topic_indicators = [
        "i want to learn", "i need", "show me a course", "teach me",
        "chahiye", "padhna hai", "want to study", "looking for"
    ]
    
    # Follow-up indicators
    followup_indicators = [
        "what", "why", "how", "explain", "tell me about",
        "kya hai", "kaise", "kyun", "matlab", "covered",
        "topics", "learn from", "outcomes"
    ]
    
    # Check if user has a recent video
    has_recent_video = session.last_video_id is not None
    
    # Check for context references (highest priority)
    has_context_ref = any(ref in question_lower for ref in context_references)
    
    # Count indicators
    new_topic_score = sum(1 for ind in new_topic_indicators if ind in question_lower)
    followup_score = sum(1 for ind in followup_indicators if ind in question_lower)
    
    # Decision logic
    # 1. If they reference "this/that course/video", it's definitely a follow-up
    if has_recent_video and has_context_ref:
        return "transcript"
    # 2. If they have a video and asking follow-up questions
    elif has_recent_video and followup_score > 0 and new_topic_score == 0:
        return "transcript"
    # 3. Strong new topic indicators
    elif new_topic_score > 0:
        return "video"
    # 4. Default based on whether they have a recent video
    else:
        return "transcript" if has_recent_video else "video"


def process_structured_response(
    structured_response: Dict[str, Any],
    source_docs: List,
    mode: str,
    session: UserSession
) -> Dict[str, Any]:
    """
    Process structured output from OpenAI API.
    
    Args:
        structured_response: Parsed Pydantic model output
        source_docs: Source documents used
        mode: Current mode
        session: User session
        
    Returns:
        Structured response dict
    """
    result = {
        "answer": None,
        "video_link": None,
        "video_id": None,
        "video_title": None,
        "status": "success"
    }
    
    if mode == "video":
        # Get video link and answer from structured output
        video_link = structured_response.get("video_link")
        video_title = structured_response.get("video_title")
        answer = structured_response.get("answer")
        
        if video_link:
            # VALIDATE: Check if this link actually exists in source documents
            all_links = []
            for doc in source_docs:
                # Extract all URLs from document content
                import re
                urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', doc.page_content)
                all_links.extend(urls)
            
            # Check if the returned link is in our source documents
            link_exists = video_link in all_links
            
            if link_exists:
                result["video_link"] = video_link
                result["video_title"] = video_title
                
                # Try to get video metadata from source docs
                video_info = extract_video_metadata(source_docs)
                result["video_id"] = video_info.get("video_id", "unknown")
                
                # Use title from structured output or fallback to metadata
                if not result["video_title"]:
                    result["video_title"] = video_info.get("title", "Course")
                
                # Update session with last video
                session.set_last_video(
                    video_id=result["video_id"],
                    video_title=result["video_title"],
                    video_link=video_link
                )
            else:
                # Link was hallucinated! Return educational fallback from answer field
                logger.warning(f"LLM hallucinated link: {video_link}. Using answer field instead.")
                result["answer"] = answer or "Let me help you with that! Could you tell me more about what you'd like to learn?"
                result["video_link"] = None
                result["video_title"] = None
        else:
            # No link returned - use the educational answer from structured response
            result["answer"] = answer or "Let me help you with that! Could you tell me more about what you'd like to learn?"
            result["video_link"] = None
            result["video_title"] = None
    
    else:
        # Transcript Q&A mode - return the answer
        result["answer"] = structured_response.get("answer")
    
    return result


def extract_video_link(text: str) -> Optional[str]:
    """Extract video URL from text"""
    # Look for URLs
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    matches = re.findall(url_pattern, text)
    
    if matches:
        # Return first URL
        return matches[0]
    
    return None


def extract_video_metadata(source_docs: List) -> Dict[str, str]:
    """Extract video metadata from source documents"""
    for doc in source_docs:
        if doc.metadata.get("content_type") == CONTENT_TYPE_VIDEO:
            # Try video_id first, then course_id as fallback
            video_id = doc.metadata.get("video_id") or doc.metadata.get("course_id", "unknown")
            return {
                "video_id": video_id,
                "title": extract_title_from_content(doc.page_content),
            }
    
    # If no video metadata found, try to extract course_id from any doc
    for doc in source_docs:
        course_id = doc.metadata.get("course_id") or doc.metadata.get("video_id")
        if course_id and course_id != "unknown":
            return {
                "video_id": course_id,
                "title": extract_title_from_content(doc.page_content),
            }
    
    return {"video_id": "unknown", "title": "Video"}


def extract_title_from_content(content: str) -> str:
    """Extract video title from content"""
    # Look for patterns like "Title:", "Video:", "Course:"
    patterns = [
        r'(?:title|video|course)\s*:\s*([^\n]+)',
        r'^([^\n]+?)\s*https?://',  # Title before URL
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # Fallback: first line
    first_line = content.split('\n')[0]
    return first_line[:50] if len(first_line) > 50 else first_line


# ============================================================================
# HEALTH CHECK
# ============================================================================

def check_rag_health() -> Dict[str, Any]:
    """Check RAG system health"""
    try:
        return {
            "faiss_index_exists": os.path.exists(os.path.join(FAISS_DIR, "index.faiss")),
            "openai_key_configured": bool(os.getenv("OPENAI_API_KEY")),
            "session_manager_stats": get_session_manager().get_stats(),
            "status": "healthy"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {"status": "unhealthy", "error": str(e)}

