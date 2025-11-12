"""
Session Management System for Per-User RAG Chains
Addresses: CRITICAL shared memory bug, user isolation, context tracking
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import threading
import logging
import uuid
from collections import OrderedDict

logger = logging.getLogger(__name__)

# ============================================================================
# SESSION DATA STRUCTURES
# ============================================================================

@dataclass
class UserSession:
    """
    Represents a single user session with their conversation history.
    """
    session_id: str
    user_name: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    
    # Conversation state
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    last_video_id: Optional[str] = None
    last_video_title: Optional[str] = None
    last_video_link: Optional[str] = None
    
    # Mode tracking
    current_mode: str = "video_search"  # "video_search" or "transcript_qa"
    
    # Memory for the QA chain (stores chat history)
    memory_buffer: List[Dict[str, str]] = field(default_factory=list)
    
    def is_expired(self, timeout_seconds: int = 3600) -> bool:
        """Check if session has expired"""
        return datetime.now() - self.last_accessed > timedelta(seconds=timeout_seconds)
    
    def update_access(self):
        """Update last accessed time"""
        self.last_accessed = datetime.now()
    
    def add_interaction(self, question: str, answer: str, video_id: Optional[str] = None):
        """Add a Q&A interaction to history"""
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "video_id": video_id,
        })
        
        # Update memory buffer for LLM context
        self.memory_buffer.append({"role": "user", "content": question})
        self.memory_buffer.append({"role": "assistant", "content": answer})
        
        # Keep only last 10 interactions (5 Q&A pairs)
        if len(self.memory_buffer) > 20:
            self.memory_buffer = self.memory_buffer[-20:]
    
    def set_last_video(self, video_id: str, video_title: str, video_link: str):
        """Track the last video recommended to user"""
        self.last_video_id = video_id
        self.last_video_title = video_title
        self.last_video_link = video_link
        self.current_mode = "transcript_qa"  # Switch to Q&A mode
    
    def reset_video_context(self):
        """Clear video context (user moved to new topic)"""
        self.last_video_id = None
        self.last_video_title = None
        self.last_video_link = None
        self.current_mode = "video_search"
    
    def get_formatted_history(self, max_interactions: int = 3) -> str:
        """Get conversation history formatted for LLM prompt"""
        if not self.conversation_history:
            return "No previous conversation."
        
        recent = self.conversation_history[-max_interactions:]
        formatted = []
        
        for interaction in recent:
            formatted.append(f"User: {interaction['question']}")
            formatted.append(f"Assistant: {interaction['answer']}")
        
        return "\n".join(formatted)


# ============================================================================
# SESSION MANAGER - Thread-safe, LRU-based
# ============================================================================

class SessionManager:
    """
    Manages user sessions with automatic cleanup and thread safety.
    
    Features:
    - Per-user session isolation (fixes shared memory bug)
    - LRU eviction when max_sessions reached
    - Automatic cleanup of expired sessions
    - Thread-safe operations
    """
    
    def __init__(self, max_sessions: int = 1000, session_timeout: int = 3600):
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout
        
        # Thread-safe session storage (LRU ordered)
        self._sessions: OrderedDict[str, UserSession] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self.total_sessions_created = 0
        self.total_sessions_expired = 0
        
        logger.info(
            f"SessionManager initialized: max_sessions={max_sessions}, "
            f"timeout={session_timeout}s"
        )
    
    def create_session(self, user_name: Optional[str] = None, session_id: Optional[str] = None) -> str:
        """
        Create a new session for a user.
        
        Args:
            user_name: Optional user name for personalization
            session_id: Optional specific session ID to use (if not provided, generates UUID)
            
        Returns:
            session_id: Unique session identifier
        """
        with self._lock:
            # Use provided session_id or generate a new UUID
            if not session_id:
                session_id = str(uuid.uuid4())
            
            session = UserSession(session_id=session_id, user_name=user_name)
            
            # LRU eviction if at capacity
            if len(self._sessions) >= self.max_sessions:
                oldest_id = next(iter(self._sessions))
                removed = self._sessions.pop(oldest_id)
                logger.info(f"Evicted oldest session: {oldest_id}")
            
            self._sessions[session_id] = session
            self.total_sessions_created += 1
            
            logger.info(f"Created session: {session_id} for user: {user_name or 'anonymous'}")
            return session_id
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """
        Retrieve a session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            UserSession or None if not found/expired
        """
        with self._lock:
            session = self._sessions.get(session_id)
            
            if not session:
                logger.debug(f"Session not found: {session_id}")
                return None
            
            # Check expiration
            if session.is_expired(self.session_timeout):
                logger.info(f"Session expired: {session_id}")
                self._sessions.pop(session_id)
                self.total_sessions_expired += 1
                return None
            
            # Update access time and move to end (LRU)
            session.update_access()
            self._sessions.move_to_end(session_id)
            
            return session
    
    def update_session(self, session_id: str, session: UserSession):
        """
        Update an existing session.
        
        Args:
            session_id: Session identifier
            session: Updated session object
        """
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id] = session
                self._sessions.move_to_end(session_id)
    
    def delete_session(self, session_id: str):
        """Delete a session"""
        with self._lock:
            if session_id in self._sessions:
                self._sessions.pop(session_id)
                logger.info(f"Deleted session: {session_id}")
    
    def cleanup_expired_sessions(self):
        """Remove all expired sessions"""
        with self._lock:
            expired_ids = [
                sid for sid, session in self._sessions.items()
                if session.is_expired(self.session_timeout)
            ]
            
            for sid in expired_ids:
                self._sessions.pop(sid)
                self.total_sessions_expired += 1
            
            if expired_ids:
                logger.info(f"Cleaned up {len(expired_ids)} expired sessions")
    
    def get_stats(self) -> Dict:
        """Get session manager statistics"""
        with self._lock:
            return {
                "active_sessions": len(self._sessions),
                "max_sessions": self.max_sessions,
                "total_created": self.total_sessions_created,
                "total_expired": self.total_sessions_expired,
            }
    
    def get_or_create_session(self, session_id: Optional[str] = None, 
                            user_name: Optional[str] = None) -> tuple[str, UserSession]:
        """
        Get existing session or create new one.
        
        Args:
            session_id: Optional existing session ID (will be used for new session if provided)
            user_name: Optional user name for new sessions
            
        Returns:
            Tuple of (session_id, session_object)
        """
        # Try to get existing session
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session_id, session
        
        # Create new session with the provided session_id (or generate if None)
        new_id = self.create_session(user_name, session_id=session_id)
        session = self.get_session(new_id)
        return new_id, session


# ============================================================================
# GLOBAL SESSION MANAGER INSTANCE
# ============================================================================

# Singleton instance
_session_manager: Optional[SessionManager] = None
_manager_lock = threading.Lock()


def get_session_manager() -> SessionManager:
    """Get or create the global session manager instance"""
    global _session_manager
    
    if _session_manager is None:
        with _manager_lock:
            if _session_manager is None:
                _session_manager = SessionManager(
                    max_sessions=1000,
                    session_timeout=3600
                )
    
    return _session_manager


# ============================================================================
# CLEANUP THREAD
# ============================================================================

def start_cleanup_thread(interval_seconds: int = 300):
    """
    Start a background thread to cleanup expired sessions.
    
    Args:
        interval_seconds: Cleanup interval (default 5 minutes)
    """
    import time
    
    def cleanup_loop():
        while True:
            time.sleep(interval_seconds)
            try:
                manager = get_session_manager()
                manager.cleanup_expired_sessions()
            except Exception as e:
                logger.error(f"Error in cleanup thread: {e}", exc_info=True)
    
    thread = threading.Thread(target=cleanup_loop, daemon=True, name="SessionCleanup")
    thread.start()
    logger.info(f"Started session cleanup thread (interval={interval_seconds}s)")

