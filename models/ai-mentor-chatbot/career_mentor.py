import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict, Generator

load_dotenv()

# Verify the key is actually loaded from .env
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

# Initialize Gemini
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    api_key=api_key,
    streaming=False  # Set to False for complete responses
)


def get_career_advice(
    student_data: Dict,
    user_query: str,
    chat_history: List[Dict] = None
) -> str:
    """
    Returns career advice from Gemini LLM.

    Args:
        student_data: dict with keys {name, cgpa, major, skills, experience}
        user_query: str - the student's question
        chat_history: list of {role, content, timestamp} dicts

    Returns:
        str - AI mentor's response
    """
    try:
        # System prompt
        system_template = """You are an expert AI Career Mentor for university students.
Guide students in:
1. Resume building and optimization
2. Interview preparation and techniques
3. Career exploration and path planning
4. Skill development and learning strategies
5. Job search guidance

Always tailor advice based on student profile (GPA, major, skills, experience).

Student Profile:
- Name: {name}
- CGPA: {gpa}
- Major: {major}
- Skills: {skills}
- Resume/Experience: {experience}

Guidelines:
1. Be encouraging but realistic.
2. If GPA < 2.5, suggest focused academic improvement strategies.
3. Reference relevant previous conversation history when applicable.
4. Use resume/experience content to suggest concrete improvements.
5. Provide actionable, specific advice with clear next steps.
6. If a skill is missing for their career goal, suggest learning resources.
7. Keep responses concise but comprehensive (2-3 paragraphs).
"""

        # Build conversation history for context
        history_text = ""
        if chat_history:
            for m in chat_history[-10:]:  # Use last 10 messages for context
                role = "Student" if m.get("role") == "user" else "Mentor"
                content = m.get("content", m.get("message", ""))
                history_text += f"{role}: {content}\n"

        # Construct the full prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("user", "{query}\n\nPrevious Conversation Context:\n" + history_text)
        ])

        # Create the chain
        chain = prompt | gemini_llm | StrOutputParser()

        # Invoke the LLM
        response = chain.invoke({
            "name": student_data.get("name", "Student"),
            "gpa": student_data.get("cgpa", 0),
            "major": student_data.get("major", "Unknown"),
            "skills": ", ".join(student_data.get("skills", [])) or "Not specified",
            "experience": student_data.get("experience", "No experience listed"),
            "query": user_query
        })

        return response

    except Exception as e:
        # Fallback response for API failures
        error_msg = str(e)

        # Check for specific error types
        if "rate limit" in error_msg.lower():
            return "[API Rate Limited] The Gemini API rate limit has been exceeded. Please try again in a few moments."
        elif "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
            return "[Auth Error] Could not authenticate with Gemini API. Please check your API key configuration."
        elif "connection" in error_msg.lower():
            return "[Connection Error] Could not connect to Gemini API. Please check your internet connection."
        else:
            return f"[Error] Could not generate response: {error_msg}. Please try asking a different question."


def get_career_advice_streaming(
    student_data: Dict,
    user_query: str,
    chat_history: List[Dict] = None
) -> Generator[str, None, None]:
    """
    Stream career advice character by character (for future WebSocket streaming).

    Args:
        student_data: dict with keys {name, cgpa, major, skills, experience}
        user_query: str - the student's question
        chat_history: list of {role, content, timestamp} dicts

    Yields:
        str - individual chunks of the response
    """
    try:
        # Get the full response first
        full_response = get_career_advice(
            student_data, user_query, chat_history)

        # Stream it character by character (or by word chunks)
        words = full_response.split()
        for i, word in enumerate(words):
            yield word + " "

    except Exception as e:
        yield f"[Error] {str(e)}"


def validate_student_data(student_data: Dict) -> tuple[bool, str]:
    """
    Validate student profile data.

    Returns:
        (is_valid: bool, error_message: str)
    """
    required_fields = ["name", "cgpa", "major", "skills"]

    for field in required_fields:
        if field not in student_data:
            return False, f"Missing required field: {field}"

    if not isinstance(student_data["cgpa"], (int, float)):
        return False, "CGPA must be a number"

    if student_data["cgpa"] < 0 or student_data["cgpa"] > 4.1:
        return False, "CGPA must be between 0 and 4.1"

    if not isinstance(student_data["skills"], list):
        return False, "Skills must be a list"

    if len(student_data["skills"]) == 0:
        return False, "At least one skill is required"

    return True, ""


def format_career_advice(response: str) -> str:
    """
    Format the career advice response for better readability.

    Adds markdown formatting if not already present.
    """
    # If response already has markdown formatting, return as is
    if "**" in response or "##" in response or "- " in response:
        return response

    # Otherwise, return with slight formatting
    return response
