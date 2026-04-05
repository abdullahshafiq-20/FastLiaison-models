# 🚀 Quick Reference Card - AI Mentor Chatbot

## Service Info
- **URL**: `http://localhost:8001`
- **Port**: 8001
- **API Docs**: `http://localhost:8001/docs`
- **Status**: `http://localhost:8001/health`

---

## 🔗 Endpoints Summary

### REST Endpoints (HTTP)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/` | Service overview & docs |
| `GET` | `/health` | Service health check |
| `POST` | `/chat/mentor` | Send chat message |
| `GET` | `/conversations/{id}` | Get conversation history |
| `DELETE` | `/conversations/{id}` | Clear conversation |

### WebSocket Endpoint

| Protocol | Endpoint | Purpose |
|----------|----------|---------|
| `ws://` | `/ws/mentor` | Real-time chat streaming |

---

## 💬 POST /chat/mentor

**Request:**
```json
{
  "student_id": "optional-uuid",
  "student_profile": {
    "name": "John Doe",
    "cgpa": 3.5,
    "major": "Computer Science",
    "skills": ["Python", "React", "SQL"],
    "experience": "6 months internship"
  },
  "message": "How do I prepare for interviews?",
  "include_history": true
}
```

**Response:**
```json
{
  "student_id": "550e8400-e29b-41d4-a716-446655440000",
  "response": "Based on your profile...",
  "timestamp": "2026-04-05T10:30:45.123456"
}
```

**Curl:**
```bash
curl -X POST http://localhost:8001/chat/mentor \
  -H "Content-Type: application/json" \
  -d '{...}'
```

---

## 📜 GET /conversations/{student_id}

**Get conversation history for a student**

```bash
curl http://localhost:8001/conversations/550e8400-e29b-41d4-a716-446655440000
```

**Response:**
```json
{
  "student_id": "550e8400-e29b-41d4-a716-446655440000",
  "student_name": "John Doe",
  "messages": [
    {
      "role": "user",
      "content": "Question 1",
      "timestamp": "2026-04-05T10:25:00"
    },
    {
      "role": "assistant",
      "content": "Answer 1",
      "timestamp": "2026-04-05T10:25:15"
    }
  ],
  "created_at": "2026-04-05T10:25:00",
  "last_updated": "2026-04-05T10:25:15"
}
```

---

## 🗑️ DELETE /conversations/{student_id}

**Clear conversation history**

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

## 🔌 WebSocket /ws/mentor

### JavaScript Connection

```javascript
const ws = new WebSocket('ws://localhost:8001/ws/mentor');

// Send message
ws.send(JSON.stringify({
  "student_id": "optional",
  "student_profile": {
    "name": "John",
    "cgpa": 3.5,
    "major": "CS",
    "skills": ["Python"],
    "experience": "..."
  },
  "message": "Your question"
}));

// Receive messages
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === 'status') {
    console.log('Status:', data.status); // "generating"
  }
  if (data.type === 'response') {
    console.log('Response:', data.response);
  }
  if (data.type === 'error') {
    console.error('Error:', data.error);
  }
};
```

### Python Connection

```python
import asyncio
import json
import websockets

async def chat():
    async with websockets.connect('ws://localhost:8001/ws/mentor') as ws:
        # Send message
        await ws.send(json.dumps({
            "student_profile": {...},
            "message": "Your question"
        }))
        
        # Receive response
        response = await ws.recv()
        print(json.loads(response))

asyncio.run(chat())
```

---

## 📊 Response Message Types

### Status Message
```json
{
  "type": "status",
  "status": "generating",
  "student_id": "...",
  "timestamp": "2026-04-05T10:30:00"
}
```

### Response Message
```json
{
  "type": "response",
  "student_id": "...",
  "response": "The mentor's full response...",
  "timestamp": "2026-04-05T10:30:15"
}
```

### Error Message
```json
{
  "type": "error",
  "error": "Error message description",
  "student_id": "..."
}
```

---

## 🧪 Testing

### Test Everything in 1 Command

```bash
cd models/ai-mentor-chatbot
python test_client.py
```

This runs 6 test scenarios automatically.

---

## 🔐 Student Profile Fields

| Field | Type | Required | Example |
|-------|------|----------|---------|
| `name` | string | ✅ | "John Doe" |
| `cgpa` | float | ✅ | 3.5 |
| `major` | string | ✅ | "Computer Science" |
| `skills` | array | ✅ | ["Python", "React"] |
| `experience` | string | ❌ | "6 months internship at XYZ" |

---

## ⏱️ Response Times

| Scenario | Time | Notes |
|----------|------|-------|
| First request | 5-30s | Gemini API latency |
| Cached question | 5-30s | Gemini is stateless |
| WebSocket status | <100ms | Instant feedback |
| History retrieval | <10ms | Local JSON file |

**Tip**: WebSocket shows "generating" status while waiting.

---

## 🛠️ Setup Checklist

- [ ] `pip install -r requirements.txt`
- [ ] Create `.env` with `GEMINI_API_KEY=your_key`
- [ ] Start service: `uvicorn main:app --reload --port 8001`
- [ ] Verify at: `http://localhost:8001/health`
- [ ] Test: `python test_client.py`
- [ ] View docs: `http://localhost:8001/docs`

---

## 🔗 Integration Methods

### Method 1: Direct WebSocket (⚡ Fastest)
```javascript
const ws = new WebSocket('ws://localhost:8001/ws/mentor');
```
✅ Real-time | ✅ Simple | ⚠️ Direct connection

### Method 2: HTTP via NestJS (🛡️ Safest)
```javascript
fetch('/api/mentor/chat', {...})
```
✅ Secure | ✅ Logged | ⚠️ Slightly slower

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Connection refused | Check port 8001 is open |
| GEMINI_API_KEY not found | Create `.env` with API key |
| Long response times | Gemini API is slow, use WebSocket for feedback |
| Conversation not saved | Check write permissions to directory |
| WebSocket connection drops | Implement reconnection logic with 3s delay |

---

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| `CHATBOT_API_GUIDE.md` | Complete API reference |
| `FRONTEND_INTEGRATION.md` | React/NestJS examples |
| `UPGRADE_SUMMARY.md` | What changed & why |
| `test_client.py` | Automated testing |

---

## 🎯 Common Tasks

### Send a message and get response
```bash
curl -X POST http://localhost:8001/chat/mentor \
  -H "Content-Type: application/json" \
  -d '{"student_profile":{...},"message":"..."}'
```

### Get all past conversations
```bash
curl http://localhost:8001/conversations/STUDENT_ID
```

### Clear a student's history
```bash
curl -X DELETE http://localhost:8001/conversations/STUDENT_ID
```

### Connect WebSocket in React
```javascript
const { messages, sendMessage } = useMentorChat();
```

---

## 📞 Quick Help

- **API Questions**: See `CHATBOT_API_GUIDE.md`
- **Integration Help**: See `FRONTEND_INTEGRATION.md`
- **Testing**: Run `python test_client.py`
- **API Browser**: Go to `http://localhost:8001/docs`

---

**Version**: 2.0.0 | **Status**: ✅ Production Ready | **Last Updated**: April 5, 2026
