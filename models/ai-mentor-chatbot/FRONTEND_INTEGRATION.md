# Frontend Integration Guide - AI Mentor Chatbot

## 🏗️ Architecture Overview

**Data Flow Pattern**:
```
React Frontend → Node.js Backend API → FastAPI Chatbot Service (port 8001)
     ↓                 ↓
    UI          Database Queries
    Messages      & Student Data
```

**Key Principles**:
- ✅ Frontend **ONLY** calls Node.js backend `/api/mentor/*` endpoints
- ✅ Backend fetches student data from its database
- ✅ Backend proxies requests to FastAPI chatbot service
- ✅ No direct connection between frontend and chatbot service
- ✅ All API keys and credentials stay on the backend

> **See [NODEJS_BACKEND_INTEGRATION.md](NODEJS_BACKEND_INTEGRATION.md) for complete backend setup with Express routes, services, and middleware.**

---

## 🚀 Frontend Setup (React)

### 1. Install Dependencies

```bash
npm install axios  # For HTTP requests to backend
```

### 2. Create a Mentor Chat Hook

**hooks/useMentorChat.js**

```javascript
import { useEffect, useState, useCallback, useRef } from 'react';
import axios from 'axios';

const BACKEND_API_URL = process.env.REACT_APP_BACKEND_API_URL || 'http://localhost:3000/api';

export const useMentorChat = (studentId) => {
  const [messages, setMessages] = useState([]);
  const [studentName, setStudentName] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [conversationHistory, setConversationHistory] = useState([]);
  const initializeRef = useRef(false);

  // Initialize session on mount - fetch student data and history
  useEffect(() => {
    if (!studentId || initializeRef.current) return;
    
    initializeRef.current = true;
    initializeSession();
  }, [studentId]);

  const initializeSession = async () => {
    try {
      setIsLoading(true);
      setError(null);

      // Call backend /api/mentor/start endpoint
      const response = await axios.post(`${BACKEND_API_URL}/mentor/start`, {
        student_id: studentId
      });

      const { student_name, conversation_history } = response.data;
      
      setStudentName(student_name);
      setConversationHistory(conversation_history || []);
      setMessages(
        (conversation_history || []).map((msg, idx) => ({
          id: `msg_${idx}`,
          role: msg.role,
          content: msg.content,
          timestamp: msg.timestamp || new Date().toISOString()
        }))
      );
    } catch (err) {
      console.error('Error initializing session:', err);
      setError(err.response?.data?.error || 'Failed to initialize mentor session');
    } finally {
      setIsLoading(false);
    }
  };

  const sendMessage = useCallback(
    async (message) => {
      if (!message.trim()) return;

      try {
        setError(null);
        
        // Add user message optimistically
        const userMsg = {
          id: `user_${Date.now()}`,
          role: 'user',
          content: message,
          timestamp: new Date().toISOString()
        };
        setMessages((prev) => [...prev, userMsg]);
        setIsLoading(true);

        // Call backend /api/mentor/message endpoint
        const response = await axios.post(`${BACKEND_API_URL}/mentor/message`, {
          student_id: studentId,
          message: message
        });

        const { response: mentorResponse, timestamp } = response.data;

        // Add assistant response
        const assistantMsg = {
          id: `asst_${Date.now()}`,
          role: 'assistant',
          content: mentorResponse,
          timestamp: timestamp || new Date().toISOString()
        };
        setMessages((prev) => [...prev, assistantMsg]);
      } catch (err) {
        console.error('Error sending message:', err);
        setError(err.response?.data?.error || 'Failed to send message');
        // Remove the last user message on error
        setMessages((prev) => prev.slice(0, -1));
      } finally {
        setIsLoading(false);
      }
    },
    [studentId]
  );

  const clearHistory = useCallback(async () => {
    try {
      await axios.delete(`${BACKEND_API_URL}/mentor/history/${studentId}`);
      setMessages([]);
      setConversationHistory([]);
      setError(null);
    } catch (err) {
      console.error('Error clearing history:', err);
      setError('Failed to clear conversation history');
    }
  }, [studentId]);

  return {
    messages,
    studentName,
    isLoading,
    error,
    sendMessage,
    clearHistory
  };
};
```

### 3. Create Chat Component

**components/MentorChat.jsx**

```jsx
import React, { useState } from 'react';
import { useMentorChat } from '../hooks/useMentorChat';
import './MentorChat.css';

export function MentorChat({ studentId }) {
  const { messages, studentName, isLoading, error, sendMessage, clearHistory } = useMentorChat(studentId);
  const [input, setInput] = useState('');

  const handleSendMessage = (e) => {
    e.preventDefault();

    if (!input.trim() || isLoading) return;

    sendMessage(input);
    setInput('');
  };

  const handleClearHistory = () => {
    if (window.confirm('Clear all conversation history?')) {
      clearHistory();
    }
  };

  return (
    <div className="mentor-chat-container">
      <div className="mentor-chat-header">
        <h2>🤖 AI Career Mentor</h2>
        {studentName && <p className="mentor-student-name">Hello, {studentName}!</p>}
        <button className="mentor-clear-btn" onClick={handleClearHistory} disabled={messages.length === 0}>
          🗑️ Clear History
        </button>
      </div>

      <div className="mentor-chat-messages">
        {messages.length === 0 ? (
          <div className="mentor-empty-state">
            <p>👋 Hi! I'm your AI Career Mentor.</p>
            <p>Ask me anything about:</p>
            <ul>
              <li>📋 Interview preparation</li>
              <li>📄 Resume optimization</li>
              <li>🛤️ Career path planning</li>
              <li>🎓 Skill development</li>
              <li>💼 Job search strategies</li>
            </ul>
            <p style={{ marginTop: '20px', fontSize: '12px', opacity: 0.7 }}>
              All your data is securely managed by our backend server.
            </p>
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              className={`mentor-message mentor-message-${msg.role}`}
            >
              <div className="mentor-message-avatar">
                {msg.role === 'user' ? '👤' : '🤖'}
              </div>
              <div className="mentor-message-content">
                <p>{msg.content}</p>
                <small className="mentor-message-time">
                  {new Date(msg.timestamp).toLocaleTimeString([], { 
                    hour: '2-digit', 
                    minute: '2-digit' 
                  })}
                </small>
              </div>
            </div>
          ))
        )}

        {isLoading && (
          <div className="mentor-message mentor-message-assistant">
            <div className="mentor-message-avatar">🤖</div>
            <div className="mentor-message-content">
              <div className="mentor-typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="mentor-error-message">
            <p>⚠️ {error}</p>
            <small>Please try again or check your connection</small>
          </div>
        )}
      </div>

      <form onSubmit={handleSendMessage} className="mentor-chat-input-form">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask your mentor a question..."
          disabled={isLoading}
          autoFocus
        />
        <button type="submit" disabled={isLoading || !input.trim()}>
          {isLoading ? '⏳ Thinking...' : '📤 Send'}
        </button>
      </form>
    </div>
  );
}
```

### 4. Environment Configuration

Create a `.env.local` file in your React root directory:

```env
REACT_APP_BACKEND_API_URL=http://localhost:3000/api
```

For production:
```env
REACT_APP_BACKEND_API_URL=https://yourdomain.com/api
```

### 5. Use in Your Page

**pages/CareerGuidance.jsx**

```jsx
import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { MentorChat } from '../components/MentorChat';
import axios from 'axios';

export function CareerGuidance() {
  const { studentId } = useParams();  // From URL: /career/:studentId
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Verify student has access (backend should validate auth)
    verifyAccess();
  }, [studentId]);

  const verifyAccess = async () => {
    try {
      setIsLoading(true);
      // Call backend health check or verify endpoint
      await axios.get(
        `${process.env.REACT_APP_BACKEND_API_URL}/mentor/health`
      );
      setIsAuthenticated(true);
    } catch (err) {
      console.error('Authentication failed:', err);
      setIsAuthenticated(false);
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return <div className="loading">Loading mentor service...</div>;
  }

  if (!isAuthenticated) {
    return (
      <div className="error-container">
        <h2>Cannot access mentor service</h2>
        <p>Please ensure you're logged in and your session is valid.</p>
      </div>
    );
  }

  return (
    <div className="career-guidance-page">
      <h1>Career Guidance & Mentoring</h1>
      <p>Your personal AI mentor is ready to help with career planning.</p>
      <MentorChat studentId={studentId} />
    </div>
  );
}
```

---

## 📊 Component File Structure

Recommended React project structure:

```
src/
├── components/
│   ├── MentorChat.jsx          # Main chat component
│   └── MentorChat.css          # Chat styles
├── hooks/
│   └── useMentorChat.js        # Custom hook (calls backend API)
├── pages/
│   └── CareerGuidance.jsx      # Page using the chat
└── .env.local                  # Environment configuration
```

### 4. Add Styling

**components/MentorChat.css**

```css
/* Main Container */
.mentor-chat-container {
  display: flex;
  flex-direction: column;
  height: 600px;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  background: #ffffff;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
}

/* Header */
.mentor-chat-header {
  padding: 16px;
  border-bottom: 1px solid #e0e0e0;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.mentor-chat-header h2 {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
}

.mentor-student-name {
  margin: 4px 0 0 0;
  font-size: 13px;
  opacity: 0.9;
}

.mentor-clear-btn {
  padding: 6px 12px;
  background: rgba(255, 255, 255, 0.2);
  color: white;
  border: 1px solid rgba(255, 255, 255, 0.3);
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
  transition: all 0.2s;
}

.mentor-clear-btn:hover:not(:disabled) {
  background: rgba(255, 255, 255, 0.3);
  border-color: white;
}

.mentor-clear-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Messages Container */
.mentor-chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  background: #fafafa;
}

/* Empty State */
.mentor-empty-state {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  height: 100%;
  color: #666;
  text-align: center;
  padding: 32px;
}

.mentor-empty-state p {
  margin: 8px 0;
  font-size: 14px;
}

.mentor-empty-state p:first-child {
  font-size: 18px;
  font-weight: 500;
  color: #333;
}

.mentor-empty-state ul {
  list-style: none;
  padding: 16px 0;
  margin: 0;
  text-align: left;
  display: inline-block;
}

.mentor-empty-state li {
  padding: 8px 0;
  font-size: 13px;
}

/* Message Styles */
.mentor-message {
  display: flex;
  gap: 12px;
  animation: slideIn 0.3s ease;
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.mentor-message-user {
  justify-content: flex-end;
}

.mentor-message-assistant {
  justify-content: flex-start;
}

.mentor-message-avatar {
  font-size: 20px;
  min-width: 28px;
  text-align: center;
  line-height: 1;
}

.mentor-message-content {
  max-width: 70%;
  padding: 12px 14px;
  border-radius: 8px;
  line-height: 1.5;
  word-wrap: break-word;
}

.mentor-message-user .mentor-message-content {
  background: #667eea;
  color: white;
  border-radius: 8px 0 8px 8px;
}

.mentor-message-assistant .mentor-message-content {
  background: white;
  border: 1px solid #e0e0e0;
  border-radius: 0 8px 8px 8px;
  color: #333;
}

.mentor-message-time {
  display: block;
  font-size: 11px;
  opacity: 0.6;
  margin-top: 4px;
}

/* Typing Indicator */
.mentor-typing-indicator {
  display: flex;
  gap: 4px;
  padding: 8px 0;
}

.mentor-typing-indicator span {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #999;
  animation: typingBounce 1.4s infinite;
}

.mentor-typing-indicator span:nth-child(2) {
  animation-delay: 0.2s;
}

.mentor-typing-indicator span:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes typingBounce {
  0%, 60%, 100% {
    transform: translateY(0);
    opacity: 0.6;
  }
  30% {
    transform: translateY(-8px);
    opacity: 1;
  }
}

/* Error Message */
.mentor-error-message {
  padding: 12px 14px;
  background: #fef1f1;
  border: 1px solid #f5a5a5;
  border-radius: 4px;
  color: #c33;
}

.mentor-error-message p {
  margin: 0;
  font-size: 14px;
  font-weight: 500;
}

.mentor-error-message small {
  display: block;
  font-size: 12px;
  opacity: 0.8;
  margin-top: 4px;
}

/* Input Form */
.mentor-chat-input-form {
  display: flex;
  gap: 8px;
  padding: 16px;
  border-top: 1px solid #e0e0e0;
  background: white;
}

.mentor-chat-input-form input {
  flex: 1;
  padding: 10px 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;
  font-family: inherit;
  transition: border-color 0.2s;
}

.mentor-chat-input-form input:focus {
  outline: none;
  border-color: #667eea;
  box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

.mentor-chat-input-form input::placeholder {
  color: #999;
}

.mentor-chat-input-form input:disabled {
  background: #f5f5f5;
  cursor: not-allowed;
  color: #999;
}

.mentor-chat-input-form button {
  padding: 10px 16px;
  background: #667eea;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.2s;
  white-space: nowrap;
}

.mentor-chat-input-form button:hover:not(:disabled) {
  background: #764ba2;
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}

.mentor-chat-input-form button:active:not(:disabled) {
  transform: translateY(0);
}

.mentor-chat-input-form button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

/* Scrollbar Styling */
.mentor-chat-messages::-webkit-scrollbar {
  width: 6px;
}

.mentor-chat-messages::-webkit-scrollbar-track {
  background: #f1f1f1;
}

.mentor-chat-messages::-webkit-scrollbar-thumb {
  background: #ccc;
  border-radius: 3px;
}

.mentor-chat-messages::-webkit-scrollbar-thumb:hover {
  background: #999;
}

/* Responsive Design */
@media (max-width: 768px) {
  .mentor-chat-container {
    height: 500px;
  }

  .mentor-message-content {
    max-width: 85%;
  }

  .mentor-chat-header {
    flex-direction: column;
    gap: 8px;
    align-items: flex-start;
  }

  .mentor-clear-btn {
    align-self: flex-end;
    margin-top: 8px;
  }
}
```

---

## 🔌 API Endpoints Called by Frontend

Your React frontend will call these **Node.js backend** endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/mentor/start` | POST | Initialize session, fetch student data from DB, return history |
| `/api/mentor/message` | POST | Process message, proxy to FastAPI chatbot, return response |
| `/api/mentor/history/:studentId` | GET | Retrieve conversation history |
| `/api/mentor/history/:studentId` | DELETE | Clear conversation history |
| `/api/mentor/health` | GET | Check backend & chatbot service health |

**Request/Response Examples**:

```javascript
// 1. Start Session
const startResponse = await axios.post(`${BACKEND_API_URL}/mentor/start`, {
  student_id: 'student_12345'
});
// Response:
// {
//   "student_name": "John Doe",
//   "student_id": "student_12345",
//   "conversation_history": [...]
// }

// 2. Send Message
const messageResponse = await axios.post(`${BACKEND_API_URL}/mentor/message`, {
  student_id: 'student_12345',
  message: 'How do I prepare for interviews?'
});
// Response:
// {
//   "response": "Great question! Here are some key strategies...",
//   "timestamp": "2024-01-15T10:30:00Z"
// }

// 3. Get History
const history = await axios.get(`${BACKEND_API_URL}/mentor/history/student_12345`);
// Response:
// {
//   "student_id": "student_12345",
//   "conversation_history": [
//     { "role": "user", "content": "...", "timestamp": "..." },
//     { "role": "assistant", "content": "...", "timestamp": "..." }
//   ]
// }
```

---

## 🔧 Backend Configuration

Ensure your Node.js backend is configured with:

```javascript
// .env file
CHATBOT_API_URL=http://localhost:8001
CHATBOT_TIMEOUT=60000  // 60 seconds
```

See [NODEJS_BACKEND_INTEGRATION.md](NODEJS_BACKEND_INTEGRATION.md) for complete backend setup including:
- Express routes implementation
- MentorService with axios calls to chatbot
- StudentService for database integration
- Authentication & error handling
- Complete working code examples

---

## 🔧 Frontend Environment Setup

Create a `.env.local` file in your React root:

```env
# Frontend - points to Node.js backend (NOT directly to chatbot service)
REACT_APP_BACKEND_API_URL=http://localhost:3000/api
```

For production:
```env
REACT_APP_BACKEND_API_URL=https://yourdomain.com/api
```

---

## 📝 Integration Checklist

Frontend implementation checklist:

- ✅ Install axios: `npm install axios`
- ✅ Create `hooks/useMentorChat.js` hook calling `/api/mentor/*` endpoints
- ✅ Create `components/MentorChat.jsx` component
- ✅ Add `components/MentorChat.css` stylesheet
- ✅ Create page/route using `<MentorChat studentId={id} />`
- ✅ Set `.env.local` with backend API URL
- ✅ Test with backend running (see [NODEJS_BACKEND_INTEGRATION.md](NODEJS_BACKEND_INTEGRATION.md))
- ✅ Verify backend can reach chatbot service at `http://localhost:8001`

---

## 🧪 Testing the Flow

```javascript
// Test complete flow: Frontend → Backend → Chatbot Service

// 1. Start session (backend fetches student data from DB)
POST http://localhost:3000/api/mentor/start
{
  "student_id": "test_student_123"
}

// 2. Send message (backend proxies to chatbot at localhost:8001)
POST http://localhost:3000/api/mentor/message
{
  "student_id": "test_student_123",
  "message": "What career paths suit someone with Python skills?"
}

// 3. Verify message was processed
GET http://localhost:3000/api/mentor/history/test_student_123

// 4. Clear session
DELETE http://localhost:3000/api/mentor/history/test_student_123
```

Use Postman or curl to test these endpoints with your backend running.

---

## 🏗️ Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                        React Frontend                                │
│                  ┌─────────────────────────┐                        │
│                  │   MentorChat Component  │                        │
│                  │  + useMentorChat Hook   │                        │
│                  └────────────┬────────────┘                        │
│                               │                                      │
│                    POST /api/mentor/start                           │
│                    POST /api/mentor/message                         │
│                    GET /api/mentor/history                          │
│                               ▼                                      │
├─────────────────────────────────────────────────────────────────────┤
│                    Node.js Backend Server                            │
│          ┌──────────────────────────────────────┐                   │
│          │  Express Routes + Services            │                  │
│          │  ├─ MentorService (axios calls)       │                  │
│          │  └─ StudentService (DB queries)       │                  │
│          └────────────┬─────────────────────────┘                   │
│                       │                                              │
│    Fetches student data from database                               │
│    Proxies requests to http://localhost:8001                        │
│                       ▼                                              │
├─────────────────────────────────────────────────────────────────────┤
│                    FastAPI Chatbot Service                           │
│          ┌──────────────────────────────────────┐                   │
│          │  POST /chat/mentor                    │                  │
│          │  GET /conversations/{student_id}     │                  │
│          │  DELETE /conversations/{student_id}  │                  │
│          │  (Calls Google Gemini API)            │                  │
│          └──────────────────────────────────────┘                   │
│                                                                      │
│  Stores conversation history in conversations.json                  │
│  Provides real-time context to LLM                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📝 Summary

**Why This Architecture?**
- ✅ **Security**: API keys and credentials stay on backend server
- ✅ **Flexibility**: Backend can add logging, rate limiting, auth middleware
- ✅ **Database Integration**: Student data fetched from backend DB, not frontend
- ✅ **Scalability**: Backend can cache, queue, or load-balance requests
- ✅ **Clear Separation**: Frontend is pure UI, backend handles business logic

**Communication**:
- Frontend ↔ Backend: HTTP/REST (documented in [NODEJS_BACKEND_INTEGRATION.md](NODEJS_BACKEND_INTEGRATION.md))
- Backend ↔ Chatbot: HTTP REST to FastAPI service at port 8001
