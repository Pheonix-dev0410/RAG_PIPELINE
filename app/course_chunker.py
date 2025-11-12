"""
Course-Based Chunking Strategy

For structured course data, this keeps each course as a single semantic unit
instead of splitting arbitrarily by character count.

Benefits:
- Complete course information stays together
- Title, URL, Description never separated
- Better retrieval quality
- Cleaner for hybrid search
"""

from langchain_core.documents import Document
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def parse_course_from_lines(lines: List[str], start_idx: int, page_number: int, source: str) -> tuple[Optional[Dict], int]:
    """
    Parse a single course from lines starting at start_idx.
    Returns (course_data, next_start_idx) or (None, next_idx) if no valid course.
    """
    course_data = {
        "metadata": {},
        "metadata_text": [],
        "transcript_text": []
    }
    
    current_field = None
    i = start_idx
    in_transcript = False
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Check if we hit the next course
        if line.startswith("Course ID:") and i > start_idx:
            # Found next course, stop here
            break
        
        # Check for field labels
        if line.startswith("Course ID:"):
            course_data["metadata"]["course_id"] = line.replace("Course ID:", "").strip()
            current_field = "course_id"
            in_transcript = False
        elif line.startswith("Title:"):
            course_data["metadata"]["course_title"] = line.replace("Title:", "").strip()
            course_data["metadata_text"].append(line)
            current_field = "title"
            in_transcript = False
        elif line.startswith("Description:"):
            desc = line.replace("Description:", "").strip()
            course_data["metadata"]["description"] = desc
            course_data["metadata_text"].append(line)
            current_field = "description"
            in_transcript = False
        elif line.startswith("URL:"):
            course_data["metadata"]["course_url"] = line.replace("URL:", "").strip()
            course_data["metadata_text"].append(line)
            current_field = "url"
            in_transcript = False
        elif line.startswith("Type:"):
            course_data["metadata"]["course_type"] = line.replace("Type:", "").strip()
            course_data["metadata_text"].append(line)
            current_field = "type"
            in_transcript = False
        elif line.startswith("Level:"):
            course_data["metadata"]["course_level"] = line.replace("Level:", "").strip()
            course_data["metadata_text"].append(line)
            current_field = "level"
            in_transcript = False
        elif line.startswith("Learning Outcomes:"):
            outcomes = line.replace("Learning Outcomes:", "").strip()
            course_data["metadata"]["learning_outcomes"] = outcomes
            course_data["metadata_text"].append(line)
            current_field = "outcomes"
            in_transcript = False
        elif line.startswith("Transcript:") or line.startswith("Transcripts:"):
            # Start of transcript section
            transcript_content = line.replace("Transcript:", "").replace("Transcripts:", "").strip()
            if transcript_content:
                course_data["transcript_text"].append(transcript_content)
            in_transcript = True
            current_field = "transcript"
        elif line:
            # Continuation of current field
            if in_transcript:
                course_data["transcript_text"].append(line)
            else:
                # Add to metadata text
                course_data["metadata_text"].append(line)
                
                # Also append to current field value if it's multi-line
                if current_field == "description" and "description" in course_data["metadata"]:
                    course_data["metadata"]["description"] += " " + line
                elif current_field == "outcomes" and "learning_outcomes" in course_data["metadata"]:
                    course_data["metadata"]["learning_outcomes"] += " " + line
        
        i += 1
    
    # Validate we found a course
    if not course_data["metadata"].get("course_id"):
        return None, i
    
    return course_data, i


def split_by_courses(text: str, page_number: int = 1, source: str = "") -> List[Document]:
    """
    Split text by course boundaries using line-by-line parsing.
    NO REGEX - just clean string matching!
    
    Args:
        text: Full page text
        page_number: Page number from PDF
        source: Source PDF path
        
    Returns:
        List of Document objects, one per course
    """
    documents = []
    
    # PyMuPDF gives better text extraction, but still normalize spaces
    # Replace multiple spaces with single space (keeps newlines)
    import re
    text = re.sub(r' +', ' ', text)  # Multiple spaces → single space
    
    # Reconstruct line breaks by looking for field markers
    # This ensures each field starts on a new line
    for field in ['Course ID:', 'Title:', 'Description:', 'URL:', 'Type:', 'Level:', 'Learning Outcomes:', 'Transcript:', 'Transcripts:']:
        text = text.replace(field, '\n' + field)
    
    # Split text into lines for parsing
    lines = text.split('\n')
    
    # Find all courses
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for course start
        if line.startswith("Course ID:"):
            course_data, next_idx = parse_course_from_lines(lines, i, page_number, source)
            
            if course_data:
                course_id = course_data["metadata"].get("course_id", f"page_{page_number}")
                
                # Build metadata text
                metadata_text = "\n".join(course_data["metadata_text"]).strip()
                if not metadata_text:
                    # Fallback: use all metadata
                    metadata_parts = []
                    for key, value in course_data["metadata"].items():
                        if key not in ["course_id", "course_url", "course_title", "course_type", "course_level"]:
                            continue
                        metadata_parts.append(f"{key}: {value}")
                    metadata_text = "\n".join(metadata_parts)
                
                # Create metadata document
                metadata_doc = Document(
                    page_content=metadata_text,
                    metadata={
                        "page": page_number,
                        "source": source,
                        "chunk_type": "course",
                        "course_id": course_id,
                        "content_type": "video_metadata",
                        **course_data["metadata"]
                    }
                )
                documents.append(metadata_doc)
                logger.debug(f"✅ Created metadata doc for {course_id}")
                
                # Create transcript document if present
                if course_data["transcript_text"]:
                    transcript_text = " ".join(course_data["transcript_text"]).strip()
                    
                    transcript_doc = Document(
                        page_content=transcript_text,
                        metadata={
                            "page": page_number,
                            "source": source,
                            "chunk_type": "transcript",
                            "course_id": course_id,
                            "video_id": course_id,
                            "content_type": "transcript",
                            "course_url": course_data["metadata"].get("course_url"),
                            "course_title": course_data["metadata"].get("course_title"),
                        }
                    )
                    documents.append(transcript_doc)
                    logger.debug(f"✅ Created transcript doc for {course_id}")
            
            i = next_idx
        else:
            i += 1
    
    if not documents:
        # No structured courses found - return as single document
        logger.warning(f"No courses found on page {page_number}. Using full text.")
        return [Document(
            page_content=text.strip(),
            metadata={
                "page": page_number,
                "source": source,
                "chunk_type": "full_text",
                "content_type": "video_metadata"
            }
        )]
    
    logger.info(f"✅ Parsed page {page_number}: {len(documents)} chunks from {len([d for d in documents if d.metadata.get('content_type') == 'video_metadata'])} courses")
    return documents


def split_transcript_by_sections(text: str, page_number: int = 1, source: str = "", 
                                 video_id: str = "unknown") -> List[Document]:
    """
    Split transcript text into logical sections (for when you add transcripts later).
    
    Looks for:
    - Timestamp markers [00:00]
    - Section headers
    - Natural breaks
    
    Args:
        text: Transcript text
        page_number: Page number
        source: Source PDF path
        video_id: Associated video ID
        
    Returns:
        List of Document objects
    """
    documents = []
    
    # Check if this is a transcript (has timestamps)
    has_timestamps = bool(re.search(r'\[\d{2}:\d{2}\]', text))
    
    if not has_timestamps:
        # No timestamps - treat as single transcript chunk
        return [Document(
            page_content=text.strip(),
            metadata={
                "page": page_number,
                "source": source,
                "chunk_type": "transcript",
                "video_id": video_id,
                "content_type": "transcript",
            }
        )]
    
    # Split by timestamp sections (every 2-3 minutes of content)
    # Pattern: [MM:SS] text until next [MM:SS]
    timestamp_pattern = r'\[(\d{2}):(\d{2})\]'
    timestamp_matches = list(re.finditer(timestamp_pattern, text))
    
    if not timestamp_matches:
        return [Document(
            page_content=text.strip(),
            metadata={
                "page": page_number,
                "source": source,
                "chunk_type": "transcript",
                "video_id": video_id,
                "content_type": "transcript",
            }
        )]
    
    # Group timestamps into ~2-minute chunks
    chunk_size_seconds = 120  # 2 minutes
    current_chunk = []
    chunk_start_time = None
    
    for i, match in enumerate(timestamp_matches):
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        timestamp_seconds = minutes * 60 + seconds
        
        # Start position and end position
        start_pos = match.start()
        if i + 1 < len(timestamp_matches):
            end_pos = timestamp_matches[i + 1].start()
        else:
            end_pos = len(text)
        
        segment_text = text[start_pos:end_pos].strip()
        
        # Initialize chunk start time
        if chunk_start_time is None:
            chunk_start_time = timestamp_seconds
        
        # Add to current chunk
        current_chunk.append(segment_text)
        
        # Check if we should finalize this chunk
        time_elapsed = timestamp_seconds - chunk_start_time
        is_last = (i == len(timestamp_matches) - 1)
        
        if time_elapsed >= chunk_size_seconds or is_last:
            # Create document from current chunk
            chunk_text = "\n".join(current_chunk)
            
            if chunk_text.strip():
                doc = Document(
                    page_content=chunk_text,
                    metadata={
                        "page": page_number,
                        "source": source,
                        "chunk_type": "transcript_section",
                        "video_id": video_id,
                        "content_type": "transcript",
                        "start_time": chunk_start_time,
                        "end_time": timestamp_seconds,
                    }
                )
                documents.append(doc)
            
            # Reset for next chunk
            current_chunk = []
            chunk_start_time = None
    
    logger.info(f"Split transcript into {len(documents)} sections")
    return documents

