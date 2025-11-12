"""
Enhanced PDF Loader with Video Metadata and Transcript Separation
Addresses: Content type tagging, optimal chunking per content type
"""

from langchain_core.documents import Document
from app.config_v2 import (
    VIDEO_METADATA_SPLITTER,
    TRANSCRIPT_SPLITTER,
    CONTENT_TYPE_VIDEO,
    CONTENT_TYPE_TRANSCRIPT
)
from app.course_chunker import split_by_courses, split_transcript_by_sections
from typing import List
import logging
import re
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Toggle course-based chunking
USE_COURSE_BASED_CHUNKING = True  # Set to False to use character-based chunking

# ============================================================================
# CONTENT TYPE DETECTION
# ============================================================================

def detect_content_type(text: str) -> str:
    """
    Detect whether content is video metadata or transcript.
    
    Heuristics:
    - Contains URLs (https://) → likely video metadata
    - Contains "Video Title:", "Link:", "Course:" → video metadata
    - Long conversational text without URLs → likely transcript
    - Contains timestamps [00:00] → transcript
    
    Args:
        text: Text content to analyze
        
    Returns:
        Content type: 'video_metadata' or 'transcript'
    """
    text_lower = text.lower()
    
    # Check for video metadata indicators
    video_indicators = [
        r'https?://',  # URLs
        r'video\s*title',
        r'course\s*:',
        r'link\s*:',
        r'topic\s*:',
        r'duration\s*:',
    ]
    
    # Check for transcript indicators
    transcript_indicators = [
        r'\[\d{2}:\d{2}\]',  # Timestamps like [00:45]
        r'\d{2}:\d{2}',  # Timestamps like 00:45
        r'transcript',
        r'(hello|hi|hey)\s+(everyone|guys|students)',  # Common video openings
    ]
    
    video_score = sum(1 for pattern in video_indicators if re.search(pattern, text_lower))
    transcript_score = sum(1 for pattern in transcript_indicators if re.search(pattern, text_lower))
    
    # URL presence is strong indicator of video metadata
    if re.search(r'https?://', text):
        return CONTENT_TYPE_VIDEO
    
    # Decide based on scores
    if video_score > transcript_score:
        return CONTENT_TYPE_VIDEO
    elif transcript_score > 0:
        return CONTENT_TYPE_TRANSCRIPT
    else:
        # Default: if short text, likely metadata; if long, likely transcript
        return CONTENT_TYPE_VIDEO if len(text) < 500 else CONTENT_TYPE_TRANSCRIPT


def extract_video_id_from_url(text: str) -> str:
    """
    Extract video ID from URL for tracking.
    
    Args:
        text: Text containing video URL
        
    Returns:
        Video ID or hash of URL
    """
    # Look for common video URL patterns
    patterns = [
        r'youtube\.com/watch\?v=([^&\s]+)',
        r'youtu\.be/([^&\s]+)',
        r'vimeo\.com/(\d+)',
        r'/video/([a-zA-Z0-9_-]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    # Fallback: hash the entire URL
    urls = re.findall(r'https?://[^\s]+', text)
    if urls:
        return str(hash(urls[0]))[:8]
    
    return "unknown"


# ============================================================================
# LOAD AND SPLIT WITH METADATA
# ============================================================================

def load_and_split_pdf(pdf_path: str) -> List[Document]:
    """
    Load PDF and split into chunks with appropriate metadata.
    
    This function:
    1. Loads PDF pages
    2. Detects content type (video metadata vs transcript)
    3. Applies appropriate chunking strategy (course-based or character-based)
    4. Adds metadata tags for filtering
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        List of Document objects with metadata
        
    Raises:
        FileNotFoundError: If PDF doesn't exist
        ValueError: If PDF is invalid or empty
    """
    try:
        logger.info(f"Loading PDF: {pdf_path}")
        logger.info(f"Course-based chunking: {'ENABLED' if USE_COURSE_BASED_CHUNKING else 'DISABLED'}")
        
        # Load PDF using PyMuPDF (better text extraction than pypdf)
        doc = fitz.open(pdf_path)
        
        if len(doc) == 0:
            raise ValueError("PDF contains no pages")
        
        # Extract text from each page and create Document objects
        pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")  # Extract plain text
            
            pages.append(Document(
                page_content=text,
                metadata={
                    "page": page_num + 1,
                    "source": pdf_path
                }
            ))
        
        doc.close()
        
        logger.info(f"Loaded {len(pages)} pages from PDF")
        
        # Process each page and detect content type
        processed_docs = []
        
        for page_num, page in enumerate(pages):
            content_type = detect_content_type(page.page_content)
            logger.debug(f"Page {page_num + 1}: Detected as {content_type}")
            
            # Apply appropriate chunking strategy
            if USE_COURSE_BASED_CHUNKING:
                # Course-based chunking
                if content_type == CONTENT_TYPE_VIDEO:
                    # Split by course boundaries
                    chunks = split_by_courses(
                        page.page_content, 
                        page_number=page_num + 1,
                        source=pdf_path
                    )
                else:
                    # Split transcripts by logical sections
                    video_id = extract_video_id_from_url(page.page_content)
                    chunks = split_transcript_by_sections(
                        page.page_content,
                        page_number=page_num + 1,
                        source=pdf_path,
                        video_id=video_id
                    )
                
                # Metadata already added by course_chunker functions
                processed_docs.extend(chunks)
                
            else:
                # Character-based chunking (original method)
                splitter = (
                    VIDEO_METADATA_SPLITTER 
                    if content_type == CONTENT_TYPE_VIDEO 
                    else TRANSCRIPT_SPLITTER
                )
                
                chunks = splitter.split_documents([page])
                
                # Add metadata to each chunk
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        "content_type": content_type,
                        "page": page_num + 1,
                        "chunk_index": i,
                        "source": pdf_path,
                    })
                    
                    # If video metadata, try to extract video ID
                    if content_type == CONTENT_TYPE_VIDEO:
                        video_id = extract_video_id_from_url(chunk.page_content)
                        chunk.metadata["video_id"] = video_id
                    
                    processed_docs.append(chunk)
        
        logger.info(
            f"Processed {len(processed_docs)} chunks: "
            f"{sum(1 for d in processed_docs if d.metadata['content_type'] == CONTENT_TYPE_VIDEO)} video metadata, "
            f"{sum(1 for d in processed_docs if d.metadata['content_type'] == CONTENT_TYPE_TRANSCRIPT)} transcript"
        )
        
        if USE_COURSE_BASED_CHUNKING:
            # Log course-specific info
            course_ids = [d.metadata.get('course_id') for d in processed_docs if d.metadata.get('course_id')]
            logger.info(f"Extracted {len(course_ids)} distinct courses")
        
        return processed_docs
        
    except FileNotFoundError as e:
        logger.error(f"PDF file not found: {pdf_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading PDF: {e}", exc_info=True)
        raise


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_documents_by_type(documents: List[Document], content_type: str) -> List[Document]:
    """
    Filter documents by content type.
    
    Args:
        documents: List of documents
        content_type: Type to filter ('video_metadata' or 'transcript')
        
    Returns:
        Filtered list of documents
    """
    return [
        doc for doc in documents 
        if doc.metadata.get("content_type") == content_type
    ]


def get_transcript_for_video(documents: List[Document], video_id: str) -> List[Document]:
    """
    Get transcript chunks for a specific video.
    
    Args:
        documents: List of all documents
        video_id: Video ID to find transcript for
        
    Returns:
        List of transcript documents for that video
    """
    # This assumes transcript chunks are on the same or adjacent pages as video metadata
    # You might need to enhance this based on your PDF structure
    return [
        doc for doc in documents
        if doc.metadata.get("content_type") == CONTENT_TYPE_TRANSCRIPT
        and (doc.metadata.get("video_id") == video_id or 
             video_id in doc.page_content[:200])  # Video ID mentioned in transcript
    ]

