# AI Career Mentor Chatbot - API Guide

## Overview

The updated chatbot service provides **two ways to interact**:
1. **HTTP Endpoints** - For simple request/response calls
2. **WebSocket** - For real-time streaming responses

Both methods automatically manage conversation history and context.

---

## 🚀 Quick Start

### Starting the Service

```bash
# Navigate to the chatbot directory
cd models/ai-mentor-chatbot

# Install dependencies (first time only)
pip install -r requirements.txt

# Start the service on port 8001
uvicorn main:app --reload --port 8001
```

Service will be available at:
- **Base URL**: `http://localhost:8001`
- **API Docs**: `http://localhost:8001/docs` (Swagger UI)
- **Health Check**: `http://localhost:8001/health`

---

## 📡 HTTP Endpoints

### 1. POST `/chat/mentor` - Send Message (HTTP)

**Best for**: Simple requests where you don't need real-time streaming.

**Request:**
```json
{
  "student_id": "optional-uuid-or-use-generated",
  "student_profile": {
    "name": "John Doe",
    "cgpa": 3.5,
    "major": "Computer Science",
    "skills": ["Python", "React", "SQL"],
    "experience": "Internship at XYZ Corp for 6 months"
  },
  "message": "How should I prepare for my first job interview?",
  "include_history": true
}
```

**Response:**
```json
{
  "student_id": "550e8400-e29b-41d4-a716-446655440000",
  "response": "Great question! Based on your profile in Computer Science with strong Python and React skills...",
  "timestamp": "2026-04-05T10:30:45.123456"
}
```

### 2. GET `/conversations/{student_id}` - Get History

**Retrieve full conversation for a student.**

**Request:**
```
GET /conversations/550e8400-e29b-41d4-a716-446655440000
```

**Response:**
```json
{
  "student_id": "550e8400-e29b-41d4-a716-446655440000",
  "student_name": "John Doe",
  "messages": [
    {
      "role": "user",
      "content": "How should I prepare for interviews?",
      "timestamp": "2026-04-05T10:25:00"
    },
    {
      "role": "assistant",
      "content": "Great question! Here's my advice...",
      "timestamp": "2026-04-05T10:25:15"
    }
  ],
  "created_at": "2026-04-05T10:25:00",
  "last_updated": "2026-04-05T10:25:15"
}
```

### 3. DELETE `/conversations/{student_id}` - Clear History

**Clear all conversation history for a student.**

```bash
curl -X DELETE http://localhost:8001/conversations/550e8400-e29b-41d4-a716-446655440000
```

**Response:**
```json
{
  "message": "Conversation cleared",
  "student_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

---

## 🔌 WebSocket Endpoint (Real-Time)

### WebSocket `/ws/mentor` - Real-Time Streaming Chat

**Best for**: Interactive applications where you want real-time updates as the mentor is "thinking".

#### Connection

```javascript
const ws = new WebSocket('ws://localhost:8001/ws/mentor');
```

#### Sending a Message

Send JSON message once connected:

```javascript
ws.send(JSON.stringify({
  "student_id": "550e8400-e29b-41d4-a716-446655440000",  // optional
  "student_profile": {
    "name": "John Doe",
    "cgpa": 3.5,
    "major": "Computer Science",
    "skills": ["Python", "React", "SQL"],
    "experience": "Internship at XYZ Corp"
  },
  "message": "How can I improve my resume?"
}));
```

#### Receiving Messages

The server sends different message types:

**Type 1: Status (generating)**
```json
{
  "type": "status",
  "status": "generating",
  "student_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2026-04-05T10:30:00"
}
```

**Type 2: Response (complete)**
```json
{
  "type": "response",
  "student_id": "550e8400-e29b-41d4-a716-446655440000",
  "response": "Based on your profile, here are my recommendations for your resume...",
  "timestamp": "2026-04-05T10:30:15"
}
```

**Type 3: Error**
```json
{
  "type": "error",
  "error": "Failed to generate response: API rate limit exceeded",
  "student_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

---

## 💻 Complete Implementation Examples

### JavaScript/Node.js - HTTP (Axios)

```javascript
import axios from 'axios';

const chatWithMentorHTTP = async () => {
  try {
    const response = await axios.post('http://localhost:8001/chat/mentor', {
      student_profile: {
        name: 'John Doe',
        cgpa: 3.5,
        major: 'Computer Science',
        skills: ['Python', 'React', 'SQL'],
        experience: 'Internship at XYZ Corp'
      },
      message: 'How can I transition into a backend role?',
      include_history: true
    });

    console.log('Mentor:', response.data.response);
    console.log('Student ID:', response.data.student_id);
  } catch (error) {
    console.error('Error:', error.response.data);
  }
};

chatWithMentorHTTP();
```

### JavaScript/React - WebSocket (Real-Time)

```jsx
import React, { useState, useEffect, useRef } from 'react';

export function ChatWithMentor() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const wsRef = useRef(null);

  useEffect(() => {
    // Connect to WebSocket
    wsRef.current = new WebSocket('ws://localhost:8001/ws/mentor');

    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'status') {
        setIsLoading(true);
      } else if (data.type === 'response') {
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: data.response,
            timestamp: data.timestamp
          }
        ]);
        setIsLoading(false);
      } else if (data.type === 'error') {
        console.error('Mentor Error:', data.error);
        setIsLoading(false);
      }
    };

    wsRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    return () => {
      wsRef.current?.close();
    };
  }, []);

  const sendMessage = (message) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        student_id: 'user-123',  // Generate or manage this
        student_profile: {
          name: 'John Doe',
          cgpa: 3.5,
          major: 'Computer Science',
          skills: ['Python', 'React', 'SQL'],
          experience: 'Internship at XYZ Corp'
        },
        message: message
      }));

      // Add user message to state
      setMessages((prev) => [
        ...prev,
        {
          role: 'user',
          content: message,
          timestamp: new Date().toISOString()
        }
      ]);
    }
  };

  return (
    <div className="chat-container">
      {messages.map((msg, idx) => (
        <div key={idx} className={`message ${msg.role}`}>
          <p>{msg.content}</p>
          <small>{new Date(msg.timestamp).toLocaleTimeString()}</small>
        </div>
      ))}
      {isLoading && <div className="loading">Mentor is thinking...</div>}
      <input
        type="text"
        placeholder="Ask your mentor..."
        onKeyPress={(e) => {
          if (e.key === 'Enter')
            sendMessage(e.target.value);
          e.target.value = '';
        }}
      />
    </div>
  );
}
```

### Python - HTTP (requests)

```python
import requests
import json

def chat_with_mentor_http():
    url = 'http://localhost:8001/chat/mentor'
    
    payload = {
        'student_profile': {
            'name': 'John Doe',
            'cgpa': 3.5,
            'major': 'Computer Science',
            'skills': ['Python', 'React', 'SQL'],
            'experience': 'Internship at XYZ Corp'
        },
        'message': 'How can I improve my problem-solving skills?',
        'include_history': True
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"Mentor: {data['response']}")
        print(f"Student ID: {data['student_id']}")
    else:
        print(f"Error: {response.status_code}")
        print(response.json())

chat_with_mentor_http()
```

### Python - WebSocket (websockets)

```python
import asyncio
import json
import websockets

async def chat_with_mentor_websocket():
    uri = 'ws://localhost:8001/ws/mentor'
    
    async with websockets.connect(uri) as websocket:
        # Send initial message
        message = {
            'student_profile': {
                'name': 'John Doe',
                'cgpa': 3.5,
                'major': 'Computer Science',
                'skills': ['Python', 'React', 'SQL'],
                'experience': 'Internship at XYZ Corp'
            },
            'message': 'What are the top 3 skills I should develop?'
        }
        
        await websocket.send(json.dumps(message))
        
        # Receive responses
        while True:
            response = await websocket.recv()
            data = json.loads(response)
            
            if data.get('type') == 'status':
                print(f"Status: {data['status']}")
            elif data.get('type') == 'response':
                print(f"Mentor: {data['response']}")
                break  # Response complete
            elif data.get('type') == 'error':
                print(f"Error: {data['error']}")
                break

asyncio.run(chat_with_mentor_websocket())
```

---

## 🔄 Integration with NestJS Backend

### HTTP Method (Simple)

```typescript
// nestjs-backend/src/mentor/mentor.service.ts
import { HttpService } from '@nestjs/axios';
import { Injectable } from '@nestjs/common';
import { firstValueFrom } from 'rxjs';

@Injectable()
export class MentorService {
  constructor(private httpService: HttpService) {}

  async chatWithMentor(studentProfile, message: string) {
    const response = await firstValueFrom(
      this.httpService.post('http://localhost:8001/chat/mentor', {
        student_profile: studentProfile,
        message: message,
        include_history: true
      })
    );

    return response.data;
  }

  async getConversationHistory(studentId: string) {
    const response = await firstValueFrom(
      this.httpService.get(`http://localhost:8001/conversations/${studentId}`)
    );

    return response.data;
  }
}
```

### WebSocket Method (Real-Time)

```typescript
// nestjs-backend/src/mentor/mentor.gateway.ts
import {
  WebSocketGateway,
  WebSocketServer,
  SubscribeMessage,
  OnGatewayConnection,
  OnGatewayDisconnect,
} from '@nestjs/websockets';
import { Server, Socket } from 'socket.io';
import * as WebSocket from 'ws';

@WebSocketGateway({ namespace: 'mentor' })
export class MentorGateway
  implements OnGatewayConnection, OnGatewayDisconnect
{
  @WebSocketServer()
  server: Server;

  private mentorWs: Map<string, WebSocket> = new Map();

  async handleConnection(client: Socket) {
    console.log(`Client connected: ${client.id}`);
  }

  handleDisconnect(client: Socket) {
    console.log(`Client disconnected: ${client.id}`);
    const ws = this.mentorWs.get(client.id);
    if (ws) {
      ws.close();
      this.mentorWs.delete(client.id);
    }
  }

  @SubscribeMessage('mentor-message')
  async handleMentorMessage(client: Socket, payload: any) {
    const mentorWsUrl = 'ws://localhost:8001/ws/mentor';
    let ws = this.mentorWs.get(client.id);

    if (!ws || ws.readyState !== WebSocket.OPEN) {
      ws = new WebSocket(mentorWsUrl);
      this.mentorWs.set(client.id, ws);
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      client.emit('mentor-response', data);
    };

    ws.onerror = (error) => {
      client.emit('mentor-error', { error: error.message });
    };

    ws.send(JSON.stringify(payload));
  }
}
```

---

## 📊 Comparison: HTTP vs WebSocket

| Feature | HTTP | WebSocket |
|---------|------|-----------|
| **When to use** | Simple requests, one-shot chats | Real-time, interactive chats |
| **Setup** | Simple - just POST | More setup, persistent connection |
| **Response Time** | Wait for full response | Get updates as they happen |
| **Data** | Request/response format | Streaming format |
| **Conversation History** | Auto-saved | Auto-saved |
| **Error Handling** | HTTP status codes | Custom error messages |
| **Scalability** | Stateless (load-balance friendly) | Stateful (requires connection management) |

---

## 🛠️ Troubleshooting

### CORS Issues

If frontend gets CORS errors, the service is configured to allow all origins. Verify:
```python
# In main.py, check CORSMiddleware
allow_origins=["*"]  # Or specify your domain
```

### Large Responses Taking Time

The Gemini API sometimes takes 5-30 seconds. For WebSocket, you'll see:
1. `type: "status"` → "generating"
2. `type: "response"` → Full response

### WebSocket Connection Drops

Implement reconnection logic in your client:

```javascript
function reconnectWebSocket() {
  ws = new WebSocket('ws://localhost:8001/ws/mentor');
  ws.onerror = () => {
    setTimeout(reconnectWebSocket, 3000);
  };
}
```

### Student ID Not Persisting

Always save the `student_id` returned from responses to track conversations:

```javascript
const firstResponse = await fetch('/chat/mentor', {...});
const { student_id } = await firstResponse.json();
localStorage.setItem('studentId', student_id);  // Save for future requests
```

---

## 📝 Notes

- **Conversation History**: All messages are automatically stored in `conversations.json`
- **Student ID**: Generate once and reuse to maintain conversation context
- **API Key**: Ensure `GEMINI_API_KEY` is set in `.env` file
- **Production**: Change `allow_origins` to your frontend domain
- **Port**: Service runs on port 8001 (not 8000)

---

## 🔗 Related Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [WebSocket Guide](https://fastapi.tiangolo.com/advanced/websockets/)
- [LangChain Docs](https://python.langchain.com/)
- [Gemini API](https://ai.google.dev/)
