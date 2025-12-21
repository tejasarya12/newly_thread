import threading
import queue
import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ThreadManager:
    def __init__(self):
        self.threads = {}
        self.queues = {}
        self.shutdown_event = threading.Event()
        self.running = False

    def get_queue(self, name: str) -> queue.Queue:
        """Get or create a queue by name"""
        if name not in self.queues:
            self.queues[name] = queue.Queue()
        return self.queues[name]

    def register_thread(self, name, target_func, queue_in=None, queue_out=None):
        """Register a thread with input/output queues"""
        q_in = self.get_queue(queue_in) if queue_in else None
        q_out = self.get_queue(queue_out) if queue_out else None
        
        self.threads[name] = {
            "target": target_func,
            "args": (self, q_in, q_out, self.shutdown_event)
        }

    def start_all(self):
        """Start all registered threads"""
        self.running = True
        self.shutdown_event.clear()
        
        for name, info in self.threads.items():
            t = threading.Thread(
                target=info["target"],
                args=info["args"],
                name=name,
                daemon=True
            )
            t.start()
            logger.info(f"Started Thread: {name}")

    def stop_all(self):
        """Signal all threads to stop"""
        self.running = False
        self.shutdown_event.set()
        time.sleep(1) # Give threads time to exit

_manager = None
def get_thread_manager():
    global _manager
    if not _manager:
        _manager = ThreadManager()
    return _manager


"""
Thread Manager - Orchestrates all 7 threads
Handles lifecycle, communication, and coordination
"""
"""
import threading
import queue
import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from config_threaded import THREADS, QUEUE_SIZES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThreadState(Enum):
   
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ThreadInfo:
   
    name: str
    thread: threading.Thread
    state: ThreadState
    queue_in: Optional[queue.Queue]
    queue_out: Optional[queue.Queue]
    error: Optional[str] = None


class ThreadManager:
    
    
    def __init__(self):
        self.threads: Dict[str, ThreadInfo] = {}
        self.running = False
        
        # Create queues for inter-thread communication
        self.queues = {
            "audio_frames": queue.Queue(maxsize=QUEUE_SIZES["audio_capture"]),
            "asr_output": queue.Queue(maxsize=QUEUE_SIZES["asr_output"]),
            "query_processed": queue.Queue(maxsize=QUEUE_SIZES["query_processing"]),
            "retrieval_results": queue.Queue(maxsize=QUEUE_SIZES["retrieval"]),
            "llm_output": queue.Queue(maxsize=QUEUE_SIZES["llm_input"]),
            "tts_audio": queue.Queue(maxsize=QUEUE_SIZES["tts_input"]),
            "playback": queue.Queue(maxsize=QUEUE_SIZES["audio_playback"]),
        }
        
        # Event for graceful shutdown
        self.shutdown_event = threading.Event()
        
        # Thread completion events
        self.events = {
            "asr_complete": threading.Event(),
            "query_complete": threading.Event(),
            "retrieval_complete": threading.Event(),
            "llm_complete": threading.Event(),
            "tts_complete": threading.Event(),
        }
        
        logger.info("ThreadManager initialized")
    
    def register_thread(
        self,
        name: str,
        target_func,
        queue_in: Optional[str] = None,
        queue_out: Optional[str] = None,
        daemon: bool = True
    ):
        
        thread = threading.Thread(
            target=target_func,
            name=name,
            daemon=daemon
        )
        
        self.threads[name] = ThreadInfo(
            name=name,
            thread=thread,
            state=ThreadState.INITIALIZING,
            queue_in=self.queues.get(queue_in) if queue_in else None,
            queue_out=self.queues.get(queue_out) if queue_out else None
        )
        
        logger.info(f"Registered thread: {name}")
    
    def start_thread(self, name: str):
        
        if name not in self.threads:
            logger.error(f"Thread {name} not found")
            return False
        
        thread_info = self.threads[name]
        try:
            thread_info.thread.start()
            thread_info.state = ThreadState.RUNNING
            logger.info(f"Started thread: {name}")
            return True
        except Exception as e:
            thread_info.state = ThreadState.ERROR
            thread_info.error = str(e)
            logger.error(f"Failed to start thread {name}: {e}")
            return False
    
    def start_all(self):
        
        logger.info("Starting all threads...")
        self.running = True
        
        for name in self.threads:
            self.start_thread(name)
        
        logger.info("All threads started")
    
    def stop_thread(self, name: str, timeout: float = 5.0):
        
        if name not in self.threads:
            logger.error(f"Thread {name} not found")
            return False
        
        thread_info = self.threads[name]
        thread_info.state = ThreadState.STOPPED
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Wait for thread to finish
        thread_info.thread.join(timeout=timeout)
        
        if thread_info.thread.is_alive():
            logger.warning(f"Thread {name} did not stop gracefully")
            return False
        
        logger.info(f"Stopped thread: {name}")
        return True
    
    def stop_all(self, timeout: float = 10.0):
        
        logger.info("Stopping all threads...")
        self.running = False
        self.shutdown_event.set()
        
        # Wait for all threads that are actually running
        for name, thread_info in self.threads.items():
            if thread_info.state == ThreadState.RUNNING:
                thread_info.thread.join(timeout=timeout)
            thread_info.state = ThreadState.STOPPED
        
        logger.info("All threads stopped")
    
    def get_status(self) -> Dict[str, Any]:
        
        status = {}
        for name, thread_info in self.threads.items():
            status[name] = {
                "state": thread_info.state.value,
                "alive": thread_info.thread.is_alive(),
                "error": thread_info.error
            }
        
        # Queue sizes
        queue_status = {}
        for queue_name, q in self.queues.items():
            queue_status[queue_name] = {
                "size": q.qsize(),
                "maxsize": q.maxsize,
                "full": q.full(),
                "empty": q.empty()
            }
        
        return {
            "threads": status,
            "queues": queue_status,
            "running": self.running
        }
    
    def get_queue(self, name: str) -> Optional[queue.Queue]:
        
        return self.queues.get(name)
    
    def is_running(self) -> bool:
        
        return self.running
    
    def is_shutdown(self) -> bool:
        
        return self.shutdown_event.is_set()
    
    def clear_event(self, name: str):
        
        if name in self.events:
            self.events[name].clear()
    
    def set_event(self, name: str):
        
        if name in self.events:
            self.events[name].set()
    
    def wait_for_event(self, name: str, timeout: Optional[float] = None) -> bool:
        
        if name in self.events:
            return self.events[name].wait(timeout=timeout)
        return False


# Global thread manager instance
_thread_manager: Optional[ThreadManager] = None


def get_thread_manager() -> ThreadManager:
    
    global _thread_manager
    if _thread_manager is None:
        _thread_manager = ThreadManager()
    return _thread_manager


def shutdown_thread_manager():
    
    global _thread_manager
    if _thread_manager is not None:
        _thread_manager.stop_all()
        _thread_manager = None

"""