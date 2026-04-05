"""
Test client for the AI Career Mentor Chatbot.
Demonstrates both HTTP and WebSocket usage.

Run this to test the chatbot endpoints:
    python test_client.py
"""

import asyncio
import json
import requests
import websockets
from typing import Optional


# ============= CONFIGURATION =============
BASE_URL = "http://localhost:8001"
WS_URL = "ws://localhost:8001"

# Mock student profile for testing
MOCK_STUDENT = {
    "name": "Ali Khan",
    "cgpa": 3.2,
    "major": "Computer Science",
    "skills": ["Python", "React", "SQL", "Docker"],
    "experience": "6 months internship at ABC Tech, worked on backend APIs"
}


# ============= HTTP TESTS =============

def test_http_single_message():
    """Test sending a single message via HTTP"""
    print("\n" + "="*60)
    print("TEST 1: HTTP - Single Message")
    print("="*60)

    try:
        payload = {
            "student_profile": MOCK_STUDENT,
            "message": "How should I prepare for my first job interview?",
            "include_history": True
        }

        print(f"Sending request to {BASE_URL}/chat/mentor")
        print(f"Student: {MOCK_STUDENT['name']}")
        print(f"Message: {payload['message']}\n")

        response = requests.post(
            f"{BASE_URL}/chat/mentor",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            print("✓ Response received successfully!")
            print(f"Student ID: {data['student_id']}")
            print(f"Timestamp: {data['timestamp']}")
            print(f"\nMentor's Response:\n{data['response']}")
            return data['student_id']
        else:
            print(f"✗ Error: {response.status_code}")
            print(response.json())
            return None

    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def test_http_multiple_messages(student_id: Optional[str] = None):
    """Test maintaining conversation history via HTTP"""
    print("\n" + "="*60)
    print("TEST 2: HTTP - Multiple Messages (Conversation History)")
    print("="*60)

    if student_id:
        print(f"Using existing Student ID: {student_id}\n")

    messages = [
        "What skills should I focus on for a backend engineer role?",
        "How can I improve my resume?",
        "What are some good projects I should build to get noticed?"
    ]

    for i, msg in enumerate(messages, 1):
        try:
            payload = {
                "student_id": student_id,
                "student_profile": MOCK_STUDENT,
                "message": msg,
                "include_history": True
            }

            print(f"\nMessage {i}: {msg}")
            response = requests.post(
                f"{BASE_URL}/chat/mentor",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                student_id = data['student_id']
                print(f"✓ Response: {data['response'][:100]}...")
            else:
                print(f"✗ Error: {response.status_code}")

        except Exception as e:
            print(f"✗ Error: {e}")

    return student_id


def test_http_get_history(student_id: str):
    """Test retrieving full conversation history via HTTP"""
    print("\n" + "="*60)
    print("TEST 3: HTTP - Retrieve Conversation History")
    print("="*60)

    try:
        print(f"Retrieving history for Student ID: {student_id}\n")

        response = requests.get(
            f"{BASE_URL}/conversations/{student_id}",
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            print(f"✓ Retrieved {len(data['messages'])} messages")
            print(f"Student: {data['student_name']}")
            print(f"Created: {data['created_at']}")
            print(f"Last Updated: {data['last_updated']}\n")

            for i, msg in enumerate(data['messages'], 1):
                role = "👤 Student" if msg['role'] == 'user' else "🤖 Mentor"
                print(f"{i}. {role}: {msg['content'][:80]}...")
                print(f"   Time: {msg['timestamp']}\n")
        else:
            print(f"✗ Error: {response.status_code}")

    except Exception as e:
        print(f"✗ Error: {e}")


def test_http_delete_history(student_id: str):
    """Test clearing conversation history via HTTP"""
    print("\n" + "="*60)
    print("TEST 4: HTTP - Delete Conversation History")
    print("="*60)

    try:
        print(f"Deleting history for Student ID: {student_id}\n")

        response = requests.delete(
            f"{BASE_URL}/conversations/{student_id}",
            timeout=10
        )

        if response.status_code == 200:
            print("✓ Conversation history deleted successfully!")
            print(f"Response: {response.json()}")
        else:
            print(f"✗ Error: {response.status_code}")

    except Exception as e:
        print(f"✗ Error: {e}")


def test_health_check():
    """Test service health check"""
    print("\n" + "="*60)
    print("HEALTH CHECK")
    print("="*60)

    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)

        if response.status_code == 200:
            data = response.json()
            print("✓ Service is healthy!")
            print(f"Status: {data['status']}")
            print(f"Service: {data['service']}")
            print(f"Version: {data['version']}")
        else:
            print(f"✗ Service unhealthy: {response.status_code}")

    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print("Make sure the service is running on port 8001")


# ============= WEBSOCKET TESTS =============

async def test_websocket_single_message():
    """Test sending a message via WebSocket"""
    print("\n" + "="*60)
    print("TEST 5: WebSocket - Real-Time Message")
    print("="*60)

    try:
        async with websockets.connect(f"{WS_URL}/ws/mentor") as websocket:
            print("✓ WebSocket connected!")

            payload = {
                "student_profile": MOCK_STUDENT,
                "message": "What's the best way to prepare a portfolio for tech interviews?"
            }

            print(f"\nSending message: {payload['message']}\n")
            await websocket.send(json.dumps(payload))

            # Receive responses
            received_student_id = None
            while True:
                response = await websocket.recv()
                data = json.loads(response)

                if data.get('type') == 'status':
                    print(f"📡 Status: {data['status']}")

                elif data.get('type') == 'response':
                    print("✓ Response received:")
                    print(f"{data['response']}\n")
                    received_student_id = data.get('student_id')
                    break

                elif data.get('type') == 'error':
                    print(f"✗ Error: {data['error']}")
                    break

            return received_student_id

    except Exception as e:
        print(f"✗ WebSocket error: {e}")
        return None


async def test_websocket_multiple_messages():
    """Test maintaining conversation history via WebSocket"""
    print("\n" + "="*60)
    print("TEST 6: WebSocket - Multiple Messages (Conversation)")
    print("="*60)

    messages = [
        "How can I transition from frontend to backend development?",
        "What programming languages should I learn?",
        "How long would it take me to be job-ready?"
    ]

    try:
        async with websockets.connect(f"{WS_URL}/ws/mentor") as websocket:
            print("✓ WebSocket connected!\n")

            for i, msg in enumerate(messages, 1):
                payload = {
                    "student_profile": MOCK_STUDENT,
                    "message": msg
                }

                print(f"Message {i}: {msg}")
                await websocket.send(json.dumps(payload))

                # Wait for response
                response_received = False
                while not response_received:
                    response = await websocket.recv()
                    data = json.loads(response)

                    if data.get('type') == 'response':
                        print(f"✓ Response: {data['response'][:100]}...\n")
                        response_received = True

                    elif data.get('type') == 'error':
                        print(f"✗ Error: {data['error']}\n")
                        response_received = True

    except Exception as e:
        print(f"✗ WebSocket error: {e}")


async def run_async_tests():
    """Run all async (WebSocket) tests"""
    await test_websocket_single_message()
    await test_websocket_multiple_messages()


# ============= MAIN TEST RUNNER =============

def main():
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  AI CAREER MENTOR CHATBOT - TEST CLIENT".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")

    # Test basic health
    test_health_check()

    # Test HTTP endpoints
    print("\n" + "🌐 TESTING HTTP ENDPOINTS" + "\n")

    student_id = test_http_single_message()

    if student_id:
        test_http_multiple_messages(student_id)
        test_http_get_history(student_id)
        test_http_delete_history(student_id)

    # Test WebSocket endpoints
    print("\n" + "🔌 TESTING WEBSOCKET ENDPOINTS" + "\n")

    try:
        asyncio.run(run_async_tests())
    except Exception as e:
        print(f"WebSocket tests failed: {e}")

    print("\n" + "="*60)
    print("TEST SUITE COMPLETED")
    print("="*60 + "\n")


if __name__ == "__main__":
    import sys

    # Check if service is running
    try:
        requests.get(f"{BASE_URL}/health", timeout=2)
    except:
        print("\n" + "="*60)
        print("⚠️  ERROR: Service not running!")
        print("="*60)
        print("\nTo start the service, run:")
        print("  cd models/ai-mentor-chatbot")
        print("  uvicorn main:app --reload --port 8001")
        print("\nThen run this test client again.")
        sys.exit(1)

    # Run tests
    main()
