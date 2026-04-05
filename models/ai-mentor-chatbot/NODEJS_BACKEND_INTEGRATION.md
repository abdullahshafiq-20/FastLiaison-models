# Node.js Backend Integration Guide - AI Mentor Chatbot

This guide shows how to integrate the AI Mentor Chatbot with your Node.js backend, where the backend manages student data, retrieves it from your database, and proxies requests to the chatbot service.

---

## 🏗️ Architecture

```
┌─────────────────────┐
│   React Frontend    │
│  (Chatbot Button)   │
└──────────┬──────────┘
           │
           │ POST /api/mentor/start
           │ (student_id only)
           ▼
┌──────────────────────────────────────┐
│   Node.js Backend (Your Server)      │
│                                      │
│  1. Get student_id from request      │
│  2. Query Database → StudentData     │
│  3. Store session with student_id    │
│  4. Return to frontend               │
└──────────┬───────────────────────────┘
           │
           │ Conversation happens via:
           │ POST /api/mentor/message
           │ (message + student_id)
           ▼
┌──────────────────────────────────────┐
│   Node.js Backend                    │
│                                      │
│  1. Get student_id + message         │
│  2. Retrieve stored StudentData      │
│  3. Call Chatbot Service             │
│  4. Return response to frontend      │
└──────────┬───────────────────────────┘
           │
           │ HTTP POST /chat/mentor
           │ (StudentData + message)
           ▼
┌──────────────────────────────────────┐
│   FastAPI Chatbot Service (8001)     │
│                                      │
│  - Processes message                 │
│  - Maintains history                 │
│  - Returns AI response               │
└──────────────────────────────────────┘
```

---

## 🛠️ Node.js Backend Implementation

### 1. Install Dependencies

```bash
npm install express axios dotenv cors
# or with yarn
yarn add express axios dotenv cors
```

### 2. Create Mentor Service Module

**src/services/mentorService.js**

```javascript
const axios = require('axios');

const CHATBOT_API_URL = process.env.CHATBOT_API_URL || 'http://localhost:8001';

class MentorService {
  /**
   * Send a message to the chatbot service
   * @param {Object} studentData - Student profile from database
   * @param {string} message - User's message
   * @param {string} studentId - Student's ID for tracking
   * @returns {Promise<{student_id, response, timestamp}>}
   */
  static async sendMessage(studentData, message, studentId) {
    try {
      const payload = {
        student_id: studentId,
        student_profile: {
          name: studentData.name,
          cgpa: studentData.cgpa,
          major: studentData.major || 'Unknown',
          skills: Array.isArray(studentData.skills) 
            ? studentData.skills 
            : (studentData.skills?.split(',') || []),
          experience: studentData.experience || null
        },
        message: message,
        include_history: true
      };

      console.log(`[MentorService] Sending message for student ${studentId}`);

      const response = await axios.post(
        `${CHATBOT_API_URL}/chat/mentor`,
        payload,
        {
          timeout: 60000, // 60 second timeout for Gemini API
          headers: {
            'Content-Type': 'application/json'
          }
        }
      );

      return {
        student_id: response.data.student_id,
        response: response.data.response,
        timestamp: response.data.timestamp
      };

    } catch (error) {
      console.error('[MentorService] Error:', error.message);
      
      if (error.response?.status === 400) {
        throw new Error('Invalid student profile data');
      } else if (error.response?.status === 500) {
        throw new Error('Chatbot service error. Please try again.');
      } else if (error.code === 'ECONNREFUSED') {
        throw new Error('Chatbot service unavailable');
      } else if (error.code === 'ECONNABORTED') {
        throw new Error('Request timeout. Please try again.');
      }
      
      throw error;
    }
  }

  /**
   * Get conversation history for a student
   * @param {string} studentId - Student's ID
   * @returns {Promise<Array>} - Array of messages
   */
  static async getConversationHistory(studentId) {
    try {
      const response = await axios.get(
        `${CHATBOT_API_URL}/conversations/${studentId}`,
        { timeout: 10000 }
      );

      return response.data.messages || [];

    } catch (error) {
      console.error('[MentorService] Error getting history:', error.message);
      
      if (error.response?.status === 404) {
        return []; // No conversation yet
      }
      
      throw error;
    }
  }

  /**
   * Clear conversation history
   * @param {string} studentId - Student's ID
   */
  static async clearConversation(studentId) {
    try {
      await axios.delete(
        `${CHATBOT_API_URL}/conversations/${studentId}`,
        { timeout: 10000 }
      );

      console.log(`[MentorService] Cleared conversation for ${studentId}`);

    } catch (error) {
      console.error('[MentorService] Error clearing history:', error.message);
      throw error;
    }
  }

  /**
   * Check chatbot service health
   * @returns {Promise<boolean>}
   */
  static async healthCheck() {
    try {
      const response = await axios.get(
        `${CHATBOT_API_URL}/health`,
        { timeout: 5000 }
      );

      return response.data.status === 'healthy';

    } catch (error) {
      console.error('[MentorService] Chatbot service unhealthy');
      return false;
    }
  }
}

module.exports = MentorService;
```

### 3. Create Database Service

**src/services/studentService.js**

```javascript
// This is a template - adjust based on your actual database

class StudentService {
  /**
   * Get student data from your database
   * @param {string} studentId - Student's ID
   * @returns {Promise<Object>} - Student profile
   */
  static async getStudentData(studentId) {
    try {
      // Example: Using your existing database connection
      // Replace with your actual database query
      
      // const student = await YourDatabase.Student.findById(studentId);
      
      // For now, here's the expected structure:
      const student = {
        id: studentId,
        name: 'John Doe',
        cgpa: 3.5,
        major: 'Computer Science',
        skills: ['Python', 'React', 'SQL'], // or comma-separated string
        experience: 'Internship at ABC Corp for 6 months',
        email: 'john@university.edu',
        batch: 2024,
        department: 'Engineering'
      };

      if (!student) {
        throw new Error(`Student ${studentId} not found`);
      }

      // Validate required fields
      this._validateStudentData(student);

      return student;

    } catch (error) {
      console.error('[StudentService] Error fetching student:', error.message);
      throw error;
    }
  }

  /**
   * Validate student data has required fields
   * @private
   */
  static _validateStudentData(student) {
    const required = ['name', 'cgpa', 'major', 'skills'];
    
    for (const field of required) {
      if (!(field in student)) {
        throw new Error(`Missing required field: ${field}`);
      }
    }

    if (typeof student.cgpa !== 'number' || student.cgpa < 0 || student.cgpa > 4.1) {
      throw new Error('Invalid CGPA value');
    }

    if (!Array.isArray(student.skills) && typeof student.skills !== 'string') {
      throw new Error('Skills must be array or comma-separated string');
    }

    if (Array.isArray(student.skills) && student.skills.length === 0) {
      throw new Error('At least one skill is required');
    }
  }
}

module.exports = StudentService;
```

### 4. Create Express Routes

**src/routes/mentor.routes.js**

```javascript
const express = require('express');
const router = express.Router();
const MentorService = require('../services/mentorService');
const StudentService = require('../services/studentService');

// Middleware to get student from request
const getStudentMiddleware = async (req, res, next) => {
  try {
    const studentId = req.params.studentId || req.body.student_id;
    
    if (!studentId) {
      return res.status(400).json({
        error: 'Student ID is required'
      });
    }

    // Get student data from your database
    const studentData = await StudentService.getStudentData(studentId);
    
    // Attach to request for use in route handlers
    req.studentData = studentData;
    req.studentId = studentId;
    
    next();

  } catch (error) {
    return res.status(404).json({
      error: error.message || 'Student not found'
    });
  }
};

/**
 * POST /api/mentor/start
 * Initialize mentor session with student data
 */
router.post('/start', async (req, res) => {
  try {
    const { student_id } = req.body;

    if (!student_id) {
      return res.status(400).json({
        error: 'student_id is required in request body'
      });
    }

    // Fetch student data from database
    const studentData = await StudentService.getStudentData(student_id);

    // Get existing conversation history (if any)
    const history = await MentorService.getConversationHistory(student_id);

    // Return student data for frontend
    res.json({
      success: true,
      student_id: student_id,
      student_name: studentData.name,
      major: studentData.major,
      conversation_history: history,
      message: 'Mentor session started. Ask your questions!'
    });

  } catch (error) {
    console.error('[Mentor Routes] Error in /start:', error.message);
    res.status(500).json({
      error: error.message || 'Failed to start mentor session'
    });
  }
});

/**
 * POST /api/mentor/message
 * Send message to mentor
 */
router.post('/message', getStudentMiddleware, async (req, res) => {
  try {
    const { message } = req.body;
    const studentData = req.studentData;
    const studentId = req.studentId;

    if (!message || message.trim() === '') {
      return res.status(400).json({
        error: 'Message cannot be empty'
      });
    }

    console.log(`[Mentor Routes] Processing message for ${studentId}`);

    // Send to chatbot service
    const mentorResponse = await MentorService.sendMessage(
      studentData,
      message,
      studentId
    );

    res.json({
      success: true,
      student_id: mentorResponse.student_id,
      response: mentorResponse.response,
      timestamp: mentorResponse.timestamp
    });

  } catch (error) {
    console.error('[Mentor Routes] Error in /message:', error.message);
    res.status(500).json({
      error: error.message || 'Failed to get mentor response'
    });
  }
});

/**
 * GET /api/mentor/history/:studentId
 * Get conversation history
 */
router.get('/history/:studentId', getStudentMiddleware, async (req, res) => {
  try {
    const studentId = req.studentId;

    const history = await MentorService.getConversationHistory(studentId);

    res.json({
      success: true,
      student_id: studentId,
      message_count: history.length,
      messages: history
    });

  } catch (error) {
    console.error('[Mentor Routes] Error in /history:', error.message);
    res.status(500).json({
      error: error.message || 'Failed to retrieve history'
    });
  }
});

/**
 * DELETE /api/mentor/history/:studentId
 * Clear conversation history
 */
router.delete('/history/:studentId', getStudentMiddleware, async (req, res) => {
  try {
    const studentId = req.studentId;

    await MentorService.clearConversation(studentId);

    res.json({
      success: true,
      message: 'Conversation history cleared',
      student_id: studentId
    });

  } catch (error) {
    console.error('[Mentor Routes] Error in DELETE /history:', error.message);
    res.status(500).json({
      error: error.message || 'Failed to clear history'
    });
  }
});

/**
 * GET /api/mentor/health
 * Check if mentor service is available
 */
router.get('/health', async (req, res) => {
  try {
    const isHealthy = await MentorService.healthCheck();

    if (isHealthy) {
      res.json({
        success: true,
        status: 'Mentor service is healthy'
      });
    } else {
      res.status(503).json({
        success: false,
        status: 'Mentor service is unavailable'
      });
    }

  } catch (error) {
    res.status(503).json({
      success: false,
      status: 'Cannot reach mentor service'
    });
  }
});

module.exports = router;
```

### 5. Setup Express App

**src/app.js**

```javascript
const express = require('express');
const cors = require('cors');
require('dotenv').config();

// Import routes
const mentorRoutes = require('./routes/mentor.routes');

const app = express();

// Middleware
app.use(cors({
  origin: process.env.FRONTEND_URL || 'http://localhost:3000',
  credentials: true
}));
app.use(express.json());

// Routes
app.use('/api/mentor', mentorRoutes);

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'Backend server is running' });
});

// Error handling
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'development' ? err.message : undefined
  });
});

module.exports = app;
```

### 6. Main Server File

**src/server.js**

```javascript
const app = require('./app');

const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
  console.log(`✅ Backend server running on port ${PORT}`);
  console.log(`📚 API Documentation: http://localhost:${PORT}/api-docs`);
  console.log(`🤖 Mentor Service: ${process.env.CHATBOT_API_URL || 'http://localhost:8001'}`);
});
```

### 7. Environment Configuration

**.env**

```env
# Server Configuration
PORT=5000
NODE_ENV=development

# Frontend URL
FRONTEND_URL=http://localhost:3000

# Chatbot Service
CHATBOT_API_URL=http://localhost:8001

# Database (adjust based on your setup)
DATABASE_URL=your_database_connection_string

# Optional: API Keys, Secrets, etc.
JWT_SECRET=your_secret_key_here
```

### 8. Package.json Scripts

```json
{
  "scripts": {
    "start": "node src/server.js",
    "dev": "nodemon src/server.js",
    "test": "jest",
    "lint": "eslint src/"
  }
}
```

---

## 💻 Frontend Implementation (React)

Now the frontend only needs to talk to your Node.js backend:

### Create Custom Hook

**hooks/useMentorChat.js**

```javascript
import { useState, useEffect, useCallback } from 'react';

export const useMentorChat = (studentId) => {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [studentName, setStudentName] = useState(null);

  // Initialize mentor session
  useEffect(() => {
    if (!studentId) return;

    const startSession = async () => {
      try {
        const response = await fetch('/api/mentor/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ student_id: studentId })
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.error || 'Failed to start session');
        }

        setStudentName(data.student_name);
        
        // Load conversation history if it exists
        if (data.conversation_history && data.conversation_history.length > 0) {
          const historyMessages = data.conversation_history.map(msg => ({
            id: `${msg.timestamp}`,
            role: msg.role,
            content: msg.content,
            timestamp: msg.timestamp
          }));
          setMessages(historyMessages);
        }

        setError(null);

      } catch (err) {
        console.error('Failed to start mentor session:', err);
        setError(err.message);
      }
    };

    startSession();
  }, [studentId]);

  const sendMessage = useCallback(
    async (message) => {
      if (!message.trim() || !studentId) return;

      // Add user message to UI immediately
      const userMessage = {
        id: `user_${Date.now()}`,
        role: 'user',
        content: message,
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, userMessage]);
      setIsLoading(true);
      setError(null);

      try {
        const response = await fetch('/api/mentor/message', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            student_id: studentId,
            message: message
          })
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.error || 'Failed to get response');
        }

        // Add mentor response to UI
        const assistantMessage = {
          id: `asst_${Date.now()}`,
          role: 'assistant',
          content: data.response,
          timestamp: data.timestamp
        };

        setMessages(prev => [...prev, assistantMessage]);

      } catch (err) {
        console.error('Error sending message:', err);
        setError(err.message);

        // Remove the user message if request failed
        setMessages(prev => prev.filter(m => m.id !== userMessage.id));

      } finally {
        setIsLoading(false);
      }
    },
    [studentId]
  );

  const clearHistory = useCallback(async () => {
    try {
      const response = await fetch(`/api/mentor/history/${studentId}`, {
        method: 'DELETE'
      });

      if (!response.ok) {
        throw new Error('Failed to clear history');
      }

      setMessages([]);
      setError(null);

    } catch (err) {
      setError(err.message);
    }
  }, [studentId]);

  return {
    messages,
    isLoading,
    error,
    studentName,
    sendMessage,
    clearHistory
  };
};
```

### Chat Component

**components/MentorChat.jsx**

```jsx
import React, { useState } from 'react';
import { useMentorChat } from '../hooks/useMentorChat';
import './MentorChat.css';

export function MentorChat({ studentId }) {
  const { messages, isLoading, error, studentName, sendMessage, clearHistory } = useMentorChat(studentId);
  const [input, setInput] = useState('');

  if (!studentId) {
    return <div className="mentor-error">No student ID provided</div>;
  }

  const handleSendMessage = (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    sendMessage(input);
    setInput('');
  };

  return (
    <div className="mentor-chat-container">
      <div className="mentor-chat-header">
        <h2>🤖 AI Career Mentor</h2>
        {studentName && <p>Chatting with {studentName}</p>}
      </div>

      <div className="mentor-chat-messages">
        {messages.length === 0 ? (
          <div className="mentor-empty-state">
            <p>👋 Hi! I'm your AI Career Mentor.</p>
            <p>Ask me anything about interviews, resume, career paths, or skill development!</p>
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
          </div>
        )}
      </div>

      <form onSubmit={handleSendMessage} className="mentor-chat-input-form">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask your mentor..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading || !input.trim()}>
          {isLoading ? '⏳' : '📤'}
        </button>
      </form>

      <div className="mentor-chat-footer">
        <button
          onClick={clearHistory}
          className="btn-clear"
          disabled={messages.length === 0}
        >
          Clear History
        </button>
      </div>
    </div>
  );
}
```

### Usage in Main Page

**pages/ChatbotPage.jsx**

```jsx
import React, { useEffect, useState } from 'react';
import { MentorChat } from '../components/MentorChat';

export function ChatbotPage() {
  const [studentId, setStudentId] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Get student ID from your auth/user context
    // This assumes you have user data available
    const user = getCurrentUser(); // your auth method
    
    if (user?.id) {
      setStudentId(user.id);
    }
    
    setLoading(false);
  }, []);

  if (loading) {
    return <div>Loading...</div>;
  }

  if (!studentId) {
    return <div>Please log in first</div>;
  }

  return (
    <div className="chatbot-page">
      <h1>Career Guidance & Mentoring</h1>
      <MentorChat studentId={studentId} />
    </div>
  );
}
```

---

## 🔄 Complete Data Flow

```
1. USER CLICKS CHATBOT BUTTON
   ↓
2. Frontend calls: POST /api/mentor/start
   {student_id: "123"}
   ↓
3. Backend:
   - Queries DB for student data
   - Gets: {name, cgpa, major, skills, experience}
   - Returns to frontend
   ↓
4. Frontend displays chat UI with student name
   ↓
5. USER TYPES MESSAGE
   ↓
6. Frontend calls: POST /api/mentor/message
   {student_id: "123", message: "..."}
   ↓
7. Backend:
   - Retrieves stored student data (from step 3)
   - Calls Chatbot Service: POST /chat/mentor
   - Sends: {student_profile: {...}, message: "..."}
   ↓
8. Chatbot processes and returns response
   ↓
9. Backend returns response to frontend
   ↓
10. Frontend displays mentor's response
    ↓
11. REPEAT FROM STEP 5 for each new message
```

---

## 🔐 Security Benefits

✅ **Database credentials** never exposed to frontend  
✅ **Student data** stays on backend  
✅ **API keys** managed by backend only  
✅ **Request validation** on backend  
✅ **Conversation tracking** secure  
✅ **Rate limiting** can be added at backend  

---

## 🛡️ Error Handling

The implementation includes handling for:

| Error | Response | Frontend |
|-------|----------|----------|
| Missing student_id | 400 Bad Request | Show error message |
| Student not found | 404 Not Found | Show "Student not found" |
| Chatbot unavailable | 503 Service Unavailable | Show "Service down" |
| Invalid message | 400 Bad Request | Show validation error |
| Timeout | 504 Gateway Timeout | Show "Request took too long" |

---

## 📊 API Contract

### Frontend to Backend

```javascript
// START SESSION
POST /api/mentor/start
{
  "student_id": "user_123"
}

Response:
{
  "success": true,
  "student_id": "user_123",
  "student_name": "John Doe",
  "major": "Computer Science",
  "conversation_history": [...],
  "message": "Mentor session started..."
}

// SEND MESSAGE
POST /api/mentor/message
{
  "student_id": "user_123",
  "message": "How do I prepare for interviews?"
}

Response:
{
  "success": true,
  "student_id": "user_123",
  "response": "Based on your profile...",
  "timestamp": "2026-04-06T10:30:45"
}
```

### Backend to Chatbot Service

```javascript
// Backend calls Chatbot Service with complete student data
POST http://localhost:8001/chat/mentor
{
  "student_id": "user_123",
  "student_profile": {
    "name": "John Doe",
    "cgpa": 3.5,
    "major": "Computer Science",
    "skills": ["Python", "React"],
    "experience": "6 month internship"
  },
  "message": "How do I prepare for interviews?",
  "include_history": true
}
```

---

## 🚀 Deployment Checklist

- [ ] Move database credentials to environment variables
- [ ] Update StudentService with actual DB queries
- [ ] Test with real student data
- [ ] Add request logging/monitoring
- [ ] Set up error tracking (Sentry, DataDog, etc.)
- [ ] Configure CORS for your frontend domain
- [ ] Add rate limiting to prevent abuse
- [ ] Set up backup for conversation history
- [ ] Test end-to-end flow
- [ ] Deploy backend to production
- [ ] Update frontend API endpoints

---

## 🧪 Testing the Integration

### Test Backend Health
```bash
curl http://localhost:5000/health
```

### Test Starting Session
```bash
curl -X POST http://localhost:5000/api/mentor/start \
  -H "Content-Type: application/json" \
  -d '{"student_id": "test_user_123"}'
```

### Test Sending Message
```bash
curl -X POST http://localhost:5000/api/mentor/message \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": "test_user_123",
    "message": "How do I improve my resume?"
  }'
```

### Test Getting History
```bash
curl http://localhost:5000/api/mentor/history/test_user_123
```

---

## 📝 Summary

**With this setup:**
- ✅ Frontend only talks to your Node.js backend
- ✅ Backend fetches student data from your database
- ✅ Backend proxies all requests to chatbot service
- ✅ Secure: API keys and DB credentials stay on backend
- ✅ Scalable: Easy to add caching, logging, analytics
- ✅ Maintainable: Centralized student/conversation management

**Data Flow**: Frontend → Node.js Backend → Chatbot Service → Gemini API

All student data is fetched from your database first, then conversations are maintained through the backend proxy. No direct frontend-to-chatbot connections needed!
