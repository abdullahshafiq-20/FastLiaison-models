from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from pathlib import Path
import warnings
import logging
warnings.filterwarnings('ignore')

from predict import EmailClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Job Email Classifier Service",
    description="BERT-based email classifier to identify job-related emails",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the directory where this file is located
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / 'best_bert_email_classifier.pth'

# Global classifier instance (lazy loaded)
classifier: Optional[EmailClassifier] = None


def get_classifier() -> EmailClassifier:
    """
    Lazy load the classifier model
    """
    global classifier
    if classifier is None:
        try:
            # Check if model file exists
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Model file not found at: {MODEL_PATH}. "
                    f"Please ensure 'best_bert_email_classifier.pth' exists in the model directory."
                )
            classifier = EmailClassifier(model_path=str(MODEL_PATH))
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to load model: {str(e)}"
            )
    return classifier


# Pydantic Models for Request/Response
class EmailRequest(BaseModel):
    email: str = Field(..., description="Email content to classify", min_length=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "Subject: Senior Software Engineer Position - Tech Corp\n\nDear Candidate,\n\nWe are excited to announce an opening for a Senior Software Engineer position at Tech Corp. We are looking for someone with 5+ years of experience in Python and machine learning."
            }
        }


class EmailPredictionResponse(BaseModel):
    prediction: str = Field(..., description="Classification result: 'job' or 'not_job'")
    confidence: float = Field(..., description="Confidence score as percentage", ge=0, le=100)
    probabilities: Dict[str, float] = Field(..., description="Probability scores for each class")
    label: int = Field(..., description="Numeric label: 0 for not_job, 1 for job")
    label_text: str = Field(..., description="Human-readable label: 'relevant' or 'not relevant'")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "job",
                "confidence": 95.23,
                "probabilities": {
                    "not_job": 4.77,
                    "job": 95.23
                },
                "label": 1,
                "label_text": "relevant"
            }
        }


@app.post("/predict", response_model=EmailPredictionResponse)
async def predict_email(request: EmailRequest):
    """
    Classify an email as job-related or not job-related using BERT model.
    """
    try:
        logger.info(f"Received prediction request with email length: {len(request.email)}")
        
        # Get classifier instance
        model = get_classifier()
        logger.info("Classifier loaded successfully")
        
        # Make prediction
        result = model.predict(request.email)
        logger.info(f"Prediction completed: {result}")
        
        # Add human-readable label_text
        # label 1 => "relevant", label 0 => "not relevant"
        label_value = result.get("label")
        result["label_text"] = "relevant" if label_value == 1 else "irrelevant"
        
        logger.info(f"Creating response with result: {result}")
        response = EmailPredictionResponse(**result)
        logger.info("Response created successfully")
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {str(e)}"
        )


@app.post("/predict/batch")
async def predict_emails_batch(emails: list[str]):
    """
    Classify multiple emails at once.
    
    Args:
        emails: List of email contents to classify
        
    Returns:
        List of prediction results
    """
    try:
        if not emails:
            raise HTTPException(
                status_code=400,
                detail="Emails list cannot be empty"
            )
        
        # Get classifier instance
        model = get_classifier()
        
        # Make batch predictions
        results = model.predict_batch(emails)
        
        return {
            "predictions": results,
            "count": len(results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during batch prediction: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify service status
    """
    try:
        # Try to get classifier to check if model is loaded
        model = get_classifier()
        return {
            "status": "healthy",
            "service": "Job Email Classifier",
            "model_loaded": model is not None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "Job Email Classifier",
            "error": str(e)
        }


@app.get("/")
async def root():
    """
    Root endpoint with API information
    """
    return {
        "message": "Welcome to Job Email Classifier Service",
        "version": "1.0.0",
        "description": "BERT-based email classifier to identify job-related emails",
        "endpoints": {
            "/predict": "POST - Classify a single email",
            "/predict/batch": "POST - Classify multiple emails",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation (Swagger UI)",
            "/redoc": "GET - Alternative API documentation (ReDoc)"
        },
        "model": "BERT-base-uncased",
        "classes": ["not_job", "job"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)