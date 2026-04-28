from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from pathlib import Path
import threading

from xai import ExplainableJobMatcher
from model_usage import JobMatcherWithFeedback
from model import (EnhancedMLModelTrainer, EnhancedFeatureEngineering,
                   EnhancedMLModelTrainerWithFeedback, HumanizedFeedbackGenerator)

# Import model getter but catch import errors
try:
    from semantic_skill_matcher import _get_model
    SEMANTIC_MATCHER_AVAILABLE = True
except ImportError as e:
    print(f"[Warning] Could not import semantic_skill_matcher: {e}")
    SEMANTIC_MATCHER_AVAILABLE = False


app = FastAPI(
    title="Explainable AI Job Matching Service",
    description="AI-powered job matching with detailed explanations",
    version="1.0.0"
)


# ─────────────────────────────────────────────────────────────────────────────
# Startup Event - Load Semantic Skill Matcher Model
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def load_skill_matcher_model():
    """
    Load the all-MiniLM-L6-v2 model on server startup (background thread).
    Non-blocking to prevent gateway startup delay.
    """
    def load_model_background():
        if not SEMANTIC_MATCHER_AVAILABLE:
            print("[Server] ⚠️  Semantic matcher not available - will use exact matching")
            return
        
        try:
            print("[Server] Loading semantic skill matcher model (background)...")
            _get_model()
            print("[Server] ✅ Semantic skill matcher model loaded successfully!")
        except Exception as e:
            print(f"[Server] ⚠️  Warning: Failed to load skill matcher model: {str(e)}")
            print(f"[Server] ⚠️  Error type: {type(e).__name__}")
            print("[Server] ⚠️  Model will be loaded on first request")
    
    # Load in background thread to prevent startup blocking
    thread = threading.Thread(target=load_model_background, daemon=True)
    thread.start()

# Pydantic Models for Request/Response

class Skill(BaseModel):
    skill_id: str
    name: str
    proficiency_level: str = Field(..., description="Beginner, Intermediate, Advanced, Expert")
    is_verified: bool = False

class Project(BaseModel):
    p_id: str
    title: str
    description: Optional[str] = None
    is_verified: bool = False
    skills: List[str] = Field(default_factory=list, description="List of skill_ids used in project")

class Course(BaseModel):
    course_id: str
    course_name: str
    grade: float = Field(..., ge=0, le=4, description="Grade on 4.0 scale")

class StudentProfile(BaseModel):
    student_id: str
    name: str
    batch: int = Field(..., description="Graduation year, e.g., 2024")
    dept_id: Optional[str] = Field(None, description="Department code/name (optional if department_name provided)")
    department_name: str
    cgpa: float = Field(..., ge=0, le=4)
    skills: List[Skill]
    projects: List[Project]
    courses: List[Course]

class RequiredSkill(BaseModel):
    skill_id: str
    name: str
    required_level: str = Field(..., description="Beginner, Intermediate, Advanced, Expert")
    is_mandatory: bool = True
    weight: float = Field(default=1.0, description="Importance weight of this skill")

class JobDescription(BaseModel):
    job_id: str
    title: str
    description: str
    company: str
    eligible_batches: List[int] = Field(..., description="Eligible graduation years")
    eligible_departments: List[str] = Field(..., description="Eligible department names or codes, e.g., CS, SE, AI")
    eligible_cgpa_min: float = Field(..., ge=0, le=4)
    required_skills: List[RequiredSkill]

class MatchRequest(BaseModel):
    student: StudentProfile
    job: JobDescription

class MatchResponse(BaseModel):
    match_score: float
    recommendation_type: str
    explanation: Dict[str, Any]
    scores_breakdown: Dict[str, Any]
    student_name: str
    job_title: str
    company: str

class PredictResponse(BaseModel):
    match_score: float
    recommendation_type: str
    component_scores: Dict[str, float]
    basic_explanation: str
    humanized_feedback: Optional[Dict[str, Any]] = None

# Mock data adapter for XAI class
class MockDataAdapter:
    """
    Adapter to make XAI class work with in-memory data instead of DB
    """
    def __init__(self, student_data: StudentProfile, job_data: JobDescription):
        self.student_data = student_data.dict()
        self.job_data = job_data.dict()
    
    def get_student_profile(self, student_id):
        return self.student_data
    
    def get_job_details(self, job_id):
        return self.job_data
    
    def get_student_skills(self):
        return self.student_data.get('skills', [])
    
    def get_job_required_skills(self):
        return self.job_data.get('required_skills', [])
    
    def get_student_projects(self):
        return self.student_data.get('projects', [])
    
    def get_project_skills(self, project_id):
        project = next(
            (p for p in self.student_data.get('projects', []) if p['p_id'] == project_id),
            None
        )
        if project:
            # Return skills based on skill_ids in project
            return [s for s in self.student_data.get('skills', []) if s['skill_id'] in project.get('skills', [])]
        return []
    
    def get_student_courses(self):
        return self.student_data.get('courses', [])
    
    def find_relevant_courses(self, courses, job_title, job_description):
        # Simple keyword matching
        keywords = set(job_title.lower().split() + job_description.lower().split())
        relevant = []
        for course in courses:
            course_words = set(course['course_name'].lower().split())
            if keywords.intersection(course_words):
                relevant.append(course)
        return relevant
    
    def calculate_average_grade(self, courses):
        if not courses:
            return 0
        return sum(c['grade'] for c in courses) / len(courses)
    
    def compare_proficiency(self, student_level, required_level):
        levels = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3, 'Expert': 4}
        student_val = levels.get(student_level, 0)
        required_val = levels.get(required_level, 0)
        
        if student_val >= required_val:
            return 1.0
        elif student_val == required_val - 1:
            return 0.7
        else:
            return 0.4
    
    def calculate_weighted_score(self, scores):
        weights = {
            'skills': 0.40,
            'education': 0.20,
            'projects': 0.15,
            'courses': 0.15,
            'cgpa': 0.10
        }
        total = sum(scores[key]['score'] * weights[key] for key in weights)
        return round(total, 2)

# Modify XAI class to accept adapter
class AdaptedExplainableJobMatcher(ExplainableJobMatcher):
    def __init__(self, student_data, job_data):
        self.adapter = MockDataAdapter(student_data, job_data)
        self.student = self.adapter.get_student_profile(student_data.student_id)
        self.job = self.adapter.get_job_details(job_data.job_id)
        self.weights = {
            'skills': 0.40,
            'education': 0.20,
            'projects': 0.15,
            'courses': 0.15,
            'cgpa': 0.10
        }
        self.graduation_year_multiplier = self.calculate_graduation_year_multiplier()
    
    # Override methods to use adapter
    def get_student_skills(self):
        return self.adapter.get_student_skills()
    
    def get_job_required_skills(self):
        return self.adapter.get_job_required_skills()
    
    def get_student_projects(self):
        return self.adapter.get_student_projects()
    
    def get_project_skills(self, project_id):
        return self.adapter.get_project_skills(project_id)
    
    def get_student_courses(self):
        return self.adapter.get_student_courses()
    
    def find_relevant_courses(self, courses, job_title, job_description):
        return self.adapter.find_relevant_courses(courses, job_title, job_description)
    
    def calculate_average_grade(self, courses):
        return self.adapter.calculate_average_grade(courses)
    
    def compare_proficiency(self, student_level, required_level):
        return self.adapter.compare_proficiency(student_level, required_level)
    
    def calculate_weighted_score(self, scores):
        return self.adapter.calculate_weighted_score(scores)

# -----------------------------------------------------------------------------
# Helpers for ML matcher endpoint
# -----------------------------------------------------------------------------

MODEL_FILENAME = "xai_gradient_boosting.pkl"
MODEL_PATH = Path(__file__).parent / MODEL_FILENAME
_ml_matcher: Optional[JobMatcherWithFeedback] = None


def get_ml_matcher() -> JobMatcherWithFeedback:
    """
    Lazy-load the ML matcher so we only hit disk the first time /predict is used.
    """
    global _ml_matcher
    if _ml_matcher is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Trained model file not found at {MODEL_PATH.name}. Please train and export the model first."
            )
        _ml_matcher = JobMatcherWithFeedback(str(MODEL_PATH), enable_feedback=True)
    return _ml_matcher


def _normalize_student(student: StudentProfile) -> Dict[str, Any]:
    """
    Convert StudentProfile into the structure expected by JobMatcherWithFeedback.
    """
    data = student.dict()
    data['dept_name'] = data.pop('department_name')

    skill_id_to_name = {}
    normalized_skills = []
    for skill in data.get('skills', []):
        normalized_skills.append({
            'skill_id': skill['skill_id'],
            'skill_name': skill['name'],
            'proficiency_level': skill['proficiency_level'],
            'is_verified': skill.get('is_verified', False)
        })
        skill_id_to_name[skill['skill_id']] = skill['name']
    data['skills'] = normalized_skills

    normalized_projects = []
    for project in data.get('projects', []):
        normalized_projects.append({
            'p_id': project['p_id'],
            'title': project['title'],
            'description': project.get('description'),
            'is_verified': project.get('is_verified', False),
            'skills': [
                skill_id_to_name.get(skill_id, str(skill_id))
                for skill_id in project.get('skills', [])
            ]
        })
    data['projects'] = normalized_projects

    normalized_courses = []
    for course in data.get('courses', []):
        normalized_courses.append({
            'course_id': course['course_id'],
            'course_name': course['course_name'],
            'gpa': course['grade'],
            'grade': course['grade']
        })
    data['courses'] = normalized_courses

    # Provide defaults for optional collections used downstream
    data.setdefault('internships', [])

    return data


def _normalize_job(job: JobDescription) -> Dict[str, Any]:
    """
    Normalize JobDescription so feature extraction can run without key errors.
    """
    data = job.dict()

    normalized_skills = []
    for skill in data.get('required_skills', []):
        normalized_skills.append({
            'skill_id': skill['skill_id'],
            'skill_name': skill['name'],
            'required_level': skill['required_level'],
            'is_mandatory': skill.get('is_mandatory', True),
            'weight': skill.get('weight', 1.0)
        })
    data['required_skills'] = normalized_skills

    # Ensure string-based departments for fuzzy matching
    data['eligible_departments'] = [
        str(dept).strip().lower() for dept in data.get('eligible_departments', [])
    ]

    # Optional fields referenced by feedback logic
    data.setdefault('min_experience_years', 0)

    return data

@app.post("/match", response_model=MatchResponse)
async def match_student_to_job(request: MatchRequest):
    """
    Match a student to a job and provide detailed explanation
    
    Returns match score, recommendation type, and comprehensive analysis
    """
    try:
        matcher = AdaptedExplainableJobMatcher(
            student_data=request.student,
            job_data=request.job
        )
        
        result = matcher.generate_match_with_explanation()
        
        return MatchResponse(
            match_score=result['match_score'],
            recommendation_type=result['recommendation_type'],
            explanation=result['explanation'],
            scores_breakdown=result['scores_breakdown'],
            student_name=request.student.name,
            job_title=request.job.title,
            company=request.job.company
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing match: {str(e)}")


@app.post("/predict", response_model=PredictResponse)
async def predict_match_score(request: MatchRequest, feedback_mode: str = 'full'):
    """
    Predict match score using the trained ML model with optional humanized feedback.

    Parameters:
        feedback_mode (str, optional): Controls the type and detail of feedback returned in the response.
            Valid values:
                - 'full': Return comprehensive, humanized feedback with detailed explanations.
                - 'quick': Return a brief summary of the match and key points.
                - 'structured': Return structured, machine-readable feedback for programmatic use.
                - 'none': Return only the match score, with no feedback or explanation.
            Default is 'full'.
    """
    import traceback
    try:
        matcher = get_ml_matcher()

        result = matcher.match_student_with_job(
            student=_normalize_student(request.student),
            job=_normalize_job(request.job),
            feedback_mode=feedback_mode
        )

        return PredictResponse(**result)
    except HTTPException:
        # Propagate FastAPI HTTPException verbatim
        raise
    except Exception as exc:
        # Log full traceback for debugging (server-side only)
        error_traceback = traceback.format_exc()
        print(f"❌ Error in /predict endpoint:")
        print(error_traceback)
        # Return user-friendly error message
        error_msg = str(exc)
        if "not found" in error_msg.lower() or "file" in error_msg.lower():
            raise HTTPException(
                status_code=503,
                detail=f"Model file issue: {error_msg}. Please ensure the model is trained and saved."
            )
        raise HTTPException(
            status_code=500, 
            detail=f"Error generating prediction: {error_msg}"
        )




@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "XAI Job Matcher"}

# For future DB integration
@app.post("/match-by-ids")
async def match_by_ids(student_id: int, job_id: int):
    """
    Future endpoint: Match using IDs with DB lookup
    Currently not implemented - requires database connection
    """
    raise HTTPException(
        status_code=501, 
        detail="Database integration not yet implemented. Use /match endpoint with full data."
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)