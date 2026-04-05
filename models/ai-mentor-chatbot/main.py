from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from career_mentor import get_career_advice
import json
from pathlib import Path
import uuid
from datetime import datetime

app = FastAPI(
    title="AI Career Mentor Chatbot",
    description="Real-time career mentoring service with streaming responses",
    version="2.0.0"
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    # Configure this to your frontend domain in production
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= DATA MODELS =============


class StudentProfile(BaseModel):
    name: str
    cgpa: float
    major: str
    skills: List[str]
    experience: Optional[str] = None


class ChatRequest(BaseModel):
    student_id: Optional[str] = None  # Optional: use for conversation tracking
    student_profile: StudentProfile
    message: str
    include_history: bool = True  # Include conversation history for context


class ChatResponse(BaseModel):
    student_id: str
    response: str
    timestamp: str


class MessageHistory(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: str


class ConversationHistory(BaseModel):
    student_id: str
    student_name: str
    messages: List[MessageHistory]
    created_at: str
    last_updated: str


# ============= CONVERSATION MANAGEMENT =============

CONVERSATIONS_FILE = Path("conversations.json")


def load_conversations() -> Dict:
    """Load all conversations from file"""
    if not CONVERSATIONS_FILE.exists():
        CONVERSATIONS_FILE.write_text("{}")
    return json.loads(CONVERSATIONS_FILE.read_text())


def save_conversations(data: Dict):
    """Save conversations to file"""
    CONVERSATIONS_FILE.write_text(json.dumps(data, indent=4))


def get_or_create_student_id() -> str:
    """Generate unique student ID if not provided"""
    return str(uuid.uuid4())


def append_message(student_id: str, role: str, content: str) -> MessageHistory:
    """Add message to conversation history"""
    data = load_conversations()
    timestamp = datetime.now().isoformat()

    if student_id not in data:
        data[student_id] = {
            "messages": [],
            "created_at": timestamp,
            "student_name": None,
            "student_profile": None
        }

    message = {
        "role": role,
        "content": content,
        "timestamp": timestamp
    }
    data[student_id]["messages"].append(message)
    save_conversations(data)
    return MessageHistory(**message)


def get_conversation_history(student_id: str, last_n: int = 10) -> List[Dict]:
    """Get the last N messages from conversation"""
    data = load_conversations()
    if student_id not in data:
        return []

    messages = data[student_id].get("messages", [])
    return messages[-last_n:]


def clear_conversation(student_id: str):
    """Clear conversation history for a student"""
    data = load_conversations()
    if student_id in data:
        del data[student_id]
        save_conversations(data)


# ============= REST API ENDPOINTS =============

@app.get("/")
async def root():
    """Service health and documentation"""
    return {
        "service": "AI Career Mentor Chatbot",
        "version": "2.0.0",
        "endpoints": {
            "http_chat": {
                "path": "/chat/mentor",
                "method": "POST",
                "description": "Send chat message (HTTP response)"
            },
            "websocket_chat": {
                "path": "/ws/mentor",
                "method": "WebSocket",
                "description": "Real-time streaming chat (WebSocket)"
            },
            "get_history": {
                "path": "/conversations/{student_id}",
                "method": "GET",
                "description": "Retrieve conversation history"
            },
            "clear_history": {
                "path": "/conversations/{student_id}",
                "method": "DELETE",
                "description": "Clear conversation for a student"
            },
            "health": {
                "path": "/health",
                "method": "GET",
                "description": "Service health check"
            }
        },
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Service health check"""
    return {
        "status": "healthy",
        "service": "AI Career Mentor",
        "version": "2.0.0"
    }


@app.post("/chat/mentor", response_model=ChatResponse)
async def chat_mentor(request: ChatRequest):
    """
    HTTP endpoint for chat requests.
    Returns complete response at once.
    """
    try:
        # Generate student ID if not provided
        student_id = request.student_id or get_or_create_student_id()

        # Convert Pydantic model to dict
        student_data = request.student_profile.dict()
        user_query = request.message

        # Get conversation history if requested
        chat_history = None
        if request.include_history:
            chat_history = get_conversation_history(student_id)

        # Get response from AI
        ai_response = get_career_advice(student_data, user_query, chat_history)

        # Store messages in conversation history
        append_message(student_id, "user", user_query)
        append_message(student_id, "assistant", ai_response)

        timestamp = datetime.now().isoformat()
        return ChatResponse(
            student_id=student_id,
            response=ai_response,
            timestamp=timestamp
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations/{student_id}", response_model=ConversationHistory)
async def get_conversation(student_id: str):
    """Get complete conversation history for a student"""
    try:
        data = load_conversations()

        if student_id not in data:
            raise HTTPException(status_code=404, detail="Student not found")

        convo_data = data[student_id]
        messages = [
            MessageHistory(**msg) for msg in convo_data.get("messages", [])
        ]

        return ConversationHistory(
            student_id=student_id,
            student_name=convo_data.get("student_name", "Unknown"),
            messages=messages,
            created_at=convo_data.get("created_at", ""),
            last_updated=messages[-1].timestamp if messages else ""
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/conversations/{student_id}")
async def delete_conversation(student_id: str):
    """Clear conversation history for a student"""
    try:
        clear_conversation(student_id)
        return {
            "message": "Conversation cleared",
            "student_id": student_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============= WEBSOCKET ENDPOINT (REAL-TIME STREAMING) =============

@app.websocket("/ws/mentor")
async def websocket_mentor(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming chat.

    Expected message format (JSON):
    {
        "student_id": "optional-uuid",
        "student_profile": {
            "name": "John Doe",
            "cgpa": 3.5,
            "major": "CS",
            "skills": ["Python", "React"],
            "experience": "..."
        },
        "message": "Your question here"
    }
    """
    await websocket.accept()

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)

            # Extract data
            student_id = message_data.get(
                "student_id") or get_or_create_student_id()
            student_profile = message_data.get("student_profile", {})
            user_message = message_data.get("message", "")

            if not user_message:
                await websocket.send_json({
                    "error": "Message cannot be empty"
                })
                continue

            if not student_profile:
                await websocket.send_json({
                    "error": "Student profile required"
                })
                continue

            # Store user message
            append_message(student_id, "user", user_message)

            # Send acknowledgment
            await websocket.send_json({
                "type": "status",
                "status": "generating",
                "student_id": student_id,
                "timestamp": datetime.now().isoformat()
            })

            try:
                # Get conversation history for context
                chat_history = get_conversation_history(student_id)

                # Get AI response
                ai_response = get_career_advice(
                    student_profile,
                    user_message,
                    chat_history
                )

                # Store assistant message
                append_message(student_id, "assistant", ai_response)

                # Send response as complete message
                # In production, you could stream character-by-character here
                await websocket.send_json({
                    "type": "response",
                    "student_id": student_id,
                    "response": ai_response,
                    "timestamp": datetime.now().isoformat()
                })

            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "error": f"Failed to generate response: {str(e)}",
                    "student_id": student_id
                })

    except WebSocketDisconnect:
        print(f"WebSocket disconnected")
    except json.JSONDecodeError:
        await websocket.send_json({
            "error": "Invalid JSON format"
        })
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
        except:
            pass


# ============= RUN SERVER =============
# Run with: uvicorn main:app --reload --port 8001
# Or: python -m uvicorn main:app --reload --port 8001
