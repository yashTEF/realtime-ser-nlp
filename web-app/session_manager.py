"""
Session management for emotion recognition.
Simplified session tracking without real-time streaming.
"""

import time
import threading


class SessionManager:
    """Manages simple session tracking for file-based predictions."""
    
    def __init__(self):
        self.sessions = {}
        self.processing_lock = threading.Lock()
    
    def create_simple_session(self):
        """
        Create a simple session for tracking file uploads.
        
        Returns:
            Dictionary with session information
        """
        try:
            with self.processing_lock:
                session_id = f"upload_{int(time.time() * 1000)}"
                self.sessions[session_id] = {
                    'start_time': time.time(),
                    'type': 'file_upload',
                    'status': 'active'
                }
            
            return {
                'success': True,
                'session_id': session_id,
                'message': 'File upload session created'
            }
        
        except Exception as e:
            return {'success': False, 'error': f'Failed to create session: {str(e)}'}
    
    def complete_session(self, session_id, prediction_result=None):
        """
        Mark a session as completed.
        
        Args:
            session_id: ID of the session to complete
            prediction_result: Optional prediction result to store
            
        Returns:
            Dictionary with completion status
        """
        try:
            with self.processing_lock:
                if session_id in self.sessions:
                    self.sessions[session_id]['status'] = 'completed'
                    self.sessions[session_id]['end_time'] = time.time()
                    if prediction_result:
                        self.sessions[session_id]['result'] = prediction_result
                    
                    return {'success': True, 'message': 'Session completed'}
                else:
                    return {'success': False, 'error': 'Session not found'}
        
        except Exception as e:
            return {'success': False, 'error': f'Failed to complete session: {str(e)}'}
    
    def get_session_info(self, session_id):
        """Get information about a specific session."""
        with self.processing_lock:
            if session_id in self.sessions:
                return self.sessions[session_id]
            return None
    
    def get_active_sessions(self):
        """Get list of all active sessions."""
        with self.processing_lock:
            active = [
                {
                    'session_id': sid,
                    'start_time': info['start_time'],
                    'type': info['type'],
                    'status': info['status']
                }
                for sid, info in self.sessions.items()
                if info['status'] == 'active'
            ]
            return active
    
    def cleanup_old_sessions(self, max_age_hours=24):
        """Remove sessions older than max_age_hours."""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        with self.processing_lock:
            sessions_to_remove = [
                sid for sid, info in self.sessions.items()
                if current_time - info['start_time'] > max_age_seconds
            ]
            
            for sid in sessions_to_remove:
                del self.sessions[sid]
            
            return len(sessions_to_remove)


# Global session manager instance
session_manager = SessionManager()
