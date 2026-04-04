"""
FastAPI Service for Video Emotion Recognition and Transcription
Processes uploaded MP4 videos and returns comprehensive emotion insights
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import tempfile
import os
import wave
import re
from typing import Dict, List, Optional
from datetime import datetime
import torch
from collections import defaultdict, Counter
import whisper
from pydantic import BaseModel
import logging
import uuid

# Directory of this script, used for temp files
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import local utilities
from utils.face_detector import FaceDetector
from utils.emotion_predictor import EmotionPredictor
from config.config import EMOTION_LABELS


face_detector = None
emotion_predictor = None
whisper_model = None

# Pydantic models for responses
class EmotionFrame(BaseModel):
    timestamp: float
    frame_number: int
    emotion: str
    confidence: float
    all_probabilities: Dict[str, float]
    faces_detected: int


class EmotionStats(BaseModel):
    emotion: str
    count: int
    percentage: float
    avg_confidence: float


class VideoInsights(BaseModel):
    video_metadata: Dict
    emotion_analysis: Dict
    transcription: Optional[Dict] = None
    processing_time: float

# Global model instances (lazy loading)
face_detector = None
emotion_predictor = None
whisper_model = None


def get_face_detector():
    """Lazy load face detector"""
    global face_detector
    if face_detector is None:
        logger.info("Loading face detector...")
        face_detector = FaceDetector()
    return face_detector


def get_emotion_predictor():
    """Lazy load emotion predictor"""
    global emotion_predictor
    if emotion_predictor is None:
        logger.info("Loading emotion predictor...")
        emotion_predictor = EmotionPredictor()
    return emotion_predictor


def get_whisper_model():
    """Lazy load Whisper model for transcription"""
    global whisper_model
    if whisper_model is None:
        logger.info("Loading Whisper model...")
        # Use base model for balance of speed and accuracy
        whisper_model = whisper.load_model("base")
        logger.info("Whisper model loaded")
    return whisper_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Pre-load all models once at startup.
    All three are held in globals and reused across every request.
    """
    logger.info("=== STARTUP: Pre-loading all models ===")
    get_face_detector()
    get_emotion_predictor()
    get_whisper_model()
    logger.info("=== STARTUP: All models ready ===")
    yield
    # Shutdown cleanup (optional)
    logger.info("=== SHUTDOWN: Releasing resources ===")



# Initialize FastAPI app
app = FastAPI(
    title="MMIA Video Emotion Recognition API",
    description="Analyze emotions in video files and transcribe audio",
    version="1.0.0",
    lifespan=lifespan          # <-- wire the lifespan in

)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_audio_from_video(video_path: str, audio_path: str) -> bool:
    """
    Extract audio from video file using moviepy 1.0.3
    
    Args:
        video_path: Path to video file
        audio_path: Path to save extracted audio
        
    Returns:
        True if audio extracted successfully, False otherwise
    """
    try:
        from moviepy.editor import VideoFileClip
        
        logger.info(f"Loading video for audio extraction: {video_path}")
        video = VideoFileClip(video_path)
        
        if video.audio is None:
            logger.warning("No audio stream found in video")
            video.close()
            return False
        
        logger.info(f"Extracting audio to: {audio_path}")
        # Write audio to file - moviepy 1.0.3 compatible
        video.audio.write_audiofile(
            audio_path,
            fps=16000,  # Sample rate
            nbytes=2,   # 16-bit
            codec='pcm_s16le',
            verbose=False,
            logger=None  # Suppress moviepy output
        )
        
        video.close()
        
        success = os.path.exists(audio_path) and os.path.getsize(audio_path) > 0
        if success:
            logger.info(f"Audio extracted successfully: {os.path.getsize(audio_path)} bytes")
        else:
            logger.warning("Audio extraction failed or resulted in empty file")
        
        return success
        
    except Exception as e:
        logger.error(f"Error extracting audio: {e}", exc_info=True)
        return False


def load_wav_as_numpy(wav_path: str) -> np.ndarray:
    """
    Load a PCM WAV file as a float32 numpy array suitable for Whisper.
    This avoids Whisper's internal load_audio which requires system ffmpeg.
    
    Args:
        wav_path: Path to WAV file (PCM 16-bit, 16kHz expected)
        
    Returns:
        Mono float32 numpy array normalized to [-1, 1]
    """
    with wave.open(wav_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)

    if sample_width == 2:  # 16-bit
        audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:  # 32-bit
        audio = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    # Convert to mono if multi-channel
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    return audio


def transcribe_audio(audio_path: str) -> Dict:
    """
    Transcribe audio using Whisper
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary with transcription results
    """
    try:
        model = get_whisper_model()
        
        logger.info(f"Starting transcription of: {audio_path}")
        
        # Load WAV as numpy array to avoid Whisper calling system ffmpeg
        audio_array = load_wav_as_numpy(audio_path)
        logger.info(f"Loaded audio: {len(audio_array)} samples ({len(audio_array)/16000:.1f}s)")
        
        result = model.transcribe(
            audio_array,
            language=None,  # Auto-detect language
            task="transcribe",
            fp16=False,  # Disable FP16 for CPU compatibility
            verbose=False
        )
        
        # Extract segments with timestamps
        segments = []
        for segment in result.get('segments', []):
            segments.append({
                'start': round(segment['start'], 2),
                'end': round(segment['end'], 2),
                'text': segment['text'].strip(),
                'confidence': round(1.0 - segment.get('no_speech_prob', 0.0), 4)
            })
        
        full_text = result.get('text', '').strip()
        language = result.get('language', 'unknown')
        transcript_words = re.findall(r"[A-Za-z0-9']+", full_text)
        word_count = len(transcript_words)
        avg_confidence = round(
            float(np.mean([seg['confidence'] for seg in segments])) if segments else 0.0, 4
        )
        speech_detected = (
            len(segments) > 0 and
            word_count >= 3 and
            avg_confidence >= 0.65
        )
        if speech_detected:
            speech_reason = "Speech detected with sufficient confidence"
        elif not segments:
            speech_reason = "No speech segments detected by ASR"
        elif word_count < 3:
            speech_reason = "Transcript too short to trust as spoken content"
        else:
            speech_reason = "Low ASR confidence; likely background noise/non-speech"
        
        # Suppress hallucinated ASR text in user-facing payload when speech is unreliable.
        public_text = full_text if speech_detected else ""
        public_segments = segments if speech_detected else []
        
        logger.info(f"Transcription complete: {len(segments)} segments, language: {language}")
        if full_text:
            logger.info(f"Transcript preview: {full_text[:100]}...")
        else:
            logger.warning("Transcription resulted in empty text (possibly no speech in audio)")
        
        return {
            'success': True,
            'full_text': public_text,
            'language': language,
            'segments': public_segments,
            'segment_count': len(public_segments),
            'word_count': word_count,
            'avg_segment_confidence': avg_confidence,
            'speech_detected': speech_detected,
            'speech_reason': speech_reason
        }
        
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'full_text': '',
            'segments': [],
            'segment_count': 0,
            'word_count': 0,
            'avg_segment_confidence': 0.0,
            'speech_detected': False,
            'speech_reason': "Transcription failed"
        }


def analyze_video_emotions(video_path: str, sample_rate: int = 1) -> Dict:
    """
    Analyze emotions in video file frame by frame
    
    Args:
        video_path: Path to video file
        sample_rate: Process every Nth frame (1 = every frame)
        
    Returns:
        Dictionary with comprehensive emotion analysis
    """
    detector = get_face_detector()
    predictor = get_emotion_predictor()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Analyzing video: {total_frames} frames, {fps} fps, {duration:.2f}s")
    
    # Storage for analysis
    frame_emotions = []
    emotion_counts = defaultdict(int)
    emotion_confidences = defaultdict(list)
    frames_with_faces = 0
    frames_without_faces = 0
    
    # Storage for gaze analysis
    gaze_pitches = []
    gaze_yaws = []
    gaze_rolls = []
    gaze_eye_contact_scores = []
    eye_contact_count = 0
    gaze_frames_count = 0
    
    frame_number = 0
    processed_frames = 0
    
    # Reset LSTM buffer at start
    predictor.reset_lstm_buffer()
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Process every Nth frame
            if frame_number % sample_rate == 0:
                timestamp = frame_number / fps if fps > 0 else 0
                # logger.debug(f"Processing frame {frame_number}/{total_frames} (timestamp: {timestamp:.2f}s)")
                
                # Detect faces and get landmarks
                faces, face_landmarks_list = detector.detect_faces_with_landmarks(frame)
                # logger.debug(f"Frame {frame_number}: Detected {len(faces)} faces, {len(face_landmarks_list)} landmark sets")
                
                if len(faces) > 0:
                    frames_with_faces += 1
                    
                    # Process first detected face
                    bbox = faces[0]
                    face_img = detector.extract_face(frame, bbox)
                    
                    # Predict emotion
                    emotion, confidence, probabilities = predictor.predict_emotion(face_img)
                    
                    # Estimate gaze and head pose using landmarks
                    gaze_data = None
                    if len(face_landmarks_list) > 0:
                        # logger.info(f"Processing gaze for frame {frame_number} - Landmarks available: {len(face_landmarks_list)}")
                        gaze_data = detector.estimate_gaze_and_pose(frame, face_landmarks_list[0], width, height)
                        # logger.info(f"Gaze data result: {gaze_data}")
                    else:
                        logger.warning(f"No landmarks available for frame {frame_number}")
                    
                    if gaze_data:
                        gaze_pitches.append(gaze_data['pitch'])
                        gaze_yaws.append(gaze_data['yaw'])
                        gaze_rolls.append(gaze_data['roll'])
                        if 'eye_contact_score' in gaze_data:
                            gaze_eye_contact_scores.append(float(gaze_data['eye_contact_score']))
                        if gaze_data.get('eye_contact', False):
                            eye_contact_count += 1
                        gaze_frames_count += 1
                        # logger.info(f"Gaze frame count updated: {gaze_frames_count}")
                    else:
                        logger.warning(f"Gaze estimation returned None for frame {frame_number}")
                    
                    # Store results
                    frame_emotions.append({
                        'timestamp': round(timestamp, 2),
                        'frame_number': frame_number,
                        'emotion': emotion,
                        'confidence': round(float(confidence), 4),
                        'all_probabilities': {
                            k: round(float(v), 4) for k, v in probabilities.items()
                        },
                        'faces_detected': len(faces),
                        'gaze': gaze_data
                    })
                    
                    # Update statistics
                    emotion_counts[emotion] += 1
                    emotion_confidences[emotion].append(confidence)
                    
                else:
                    frames_without_faces += 1
                    # Add frame with no detection
                    frame_emotions.append({
                        'timestamp': round(timestamp, 2),
                        'frame_number': frame_number,
                        'emotion': 'No Face Detected',
                        'confidence': 0.0,
                        'all_probabilities': {},
                        'faces_detected': 0
                    })
                
                processed_frames += 1
            
            frame_number += 1
            
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    logger.info(f"Processed {processed_frames} frames, {frames_with_faces} with faces")
    
    # Calculate statistics
    total_detected_frames = sum(emotion_counts.values())
    
    emotion_statistics = []
    for emotion in EMOTION_LABELS.values():
        count = emotion_counts.get(emotion, 0)
        percentage = (count / total_detected_frames * 100) if total_detected_frames > 0 else 0
        avg_conf = np.mean(emotion_confidences.get(emotion, [0])) if emotion in emotion_confidences else 0
        
        emotion_statistics.append({
            'emotion': emotion,
            'count': count,
            'percentage': round(percentage, 2),
            'avg_confidence': round(float(avg_conf), 4)
        })
    
    # Sort by count descending
    emotion_statistics.sort(key=lambda x: x['count'], reverse=True)
    
    # Find dominant emotion
    dominant_emotion = None
    dominant_confidence = 0.0
    if emotion_statistics and emotion_statistics[0]['count'] > 0:
        dominant_emotion = emotion_statistics[0]['emotion']
        dominant_confidence = emotion_statistics[0]['avg_confidence']
    
    # Calculate overall confidence (weighted average)
    total_confidence = sum(
        count * np.mean(emotion_confidences[emotion])
        for emotion, count in emotion_counts.items()
        if emotion in emotion_confidences
    )
    overall_confidence = total_confidence / total_detected_frames if total_detected_frames > 0 else 0
    
    # Emotion timeline analysis (detect emotion changes)
    emotion_segments = []
    if frame_emotions:
        current_emotion = None
        segment_start = 0
        
        for i, frame_data in enumerate(frame_emotions):
            if frame_data['emotion'] != current_emotion and frame_data['emotion'] != 'No Face Detected':
                if current_emotion is not None:
                    emotion_segments.append({
                        'emotion': current_emotion,
                        'start_time': round(frame_emotions[segment_start]['timestamp'], 2),
                        'end_time': round(frame_emotions[i-1]['timestamp'], 2),
                        'duration': round(frame_emotions[i-1]['timestamp'] - frame_emotions[segment_start]['timestamp'], 2)
                    })
                current_emotion = frame_data['emotion']
                segment_start = i
        
        # Add final segment
        if current_emotion is not None and current_emotion != 'No Face Detected':
            emotion_segments.append({
                'emotion': current_emotion,
                'start_time': round(frame_emotions[segment_start]['timestamp'], 2),
                'end_time': round(frame_emotions[-1]['timestamp'], 2),
                'duration': round(frame_emotions[-1]['timestamp'] - frame_emotions[segment_start]['timestamp'], 2)
            })
    
    # Calculate gaze analysis
    gaze_analysis = None
    if gaze_frames_count > 0:
        gaze_analysis = {
            'average_pitch': round(float(np.mean(gaze_pitches)), 2) if gaze_pitches else 0.0,
            'average_yaw': round(float(np.mean(gaze_yaws)), 2) if gaze_yaws else 0.0,
            'average_roll': round(float(np.mean(gaze_rolls)), 2) if gaze_rolls else 0.0,
            'average_eye_contact_score': round(float(np.mean(gaze_eye_contact_scores)), 3) if gaze_eye_contact_scores else None,
            'eye_contact_percentage': round((eye_contact_count / gaze_frames_count * 100), 2),
            'gaze_frames_analyzed': gaze_frames_count,
            'eye_contact_frames': eye_contact_count
        }
        logger.info(f"✓ GAZE ANALYSIS COMPLETE: {gaze_analysis}")
    else:
        logger.warning(f"⚠ NO GAZE DATA: gaze_frames_count = {gaze_frames_count}, gaze_pitches = {len(gaze_pitches)}, gaze_yaws = {len(gaze_yaws)}")
    
    logger.info(f"DEBUG: About to return emotion_results with gaze_analysis = {gaze_analysis}")
    
    emotion_results = {
        'video_metadata': {
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'fps': round(fps, 2),
            'duration_seconds': round(duration, 2),
            'resolution': f"{width}x{height}",
            'sample_rate': sample_rate
        },
        'detection_summary': {
            'frames_with_faces': frames_with_faces,
            'frames_without_faces': frames_without_faces,
            'face_detection_rate': round(frames_with_faces / processed_frames * 100, 2) if processed_frames > 0 else 0
        },
        'emotion_analysis': {
            'dominant_emotion': dominant_emotion,
            'dominant_confidence': round(float(dominant_confidence), 4),
            'overall_confidence': round(float(overall_confidence), 4),
            'total_detections': total_detected_frames,
            'emotion_statistics': emotion_statistics,
            'emotion_segments': emotion_segments[:20]  # Limit to first 20 segments
        },
        'gaze_analysis': gaze_analysis,
        'frame_by_frame': frame_emotions[:100]  # Limit to first 100 frames for response size
    }
    
    # Verify gaze_analysis is in emotion_results
    logger.info(f"DEBUG: emotion_results keys = {list(emotion_results.keys())}")
    logger.info(f"DEBUG: emotion_results['gaze_analysis'] = {emotion_results.get('gaze_analysis', 'KEY NOT FOUND!')}")
    
    return emotion_results


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "MMIA Video Emotion Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "/analyze": "POST - Analyze video for emotions",
            "/analyze-with-transcription": "POST - Analyze video with audio transcription",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "face_detector": face_detector is not None,
            "emotion_predictor": emotion_predictor is not None,
            "whisper": whisper_model is not None
        }
    }


@app.post("/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    sample_rate: int = 1,
    include_frames: bool = False
):
    """
    Analyze emotions in uploaded video
    
    Args:
        file: MP4 video file
        sample_rate: Process every Nth frame (default: 1)
        include_frames: Include frame-by-frame data in response (default: False)
        
    Returns:
        Comprehensive emotion analysis
    """
    start_time = datetime.now()
    
    logger.info(f"Received video analysis request: {file.filename}")
    
    # Validate file type
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload MP4, AVI, MOV, or MKV video."
        )
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        # Analyze video
        results = analyze_video_emotions(tmp_path, sample_rate=sample_rate)
        
        # Remove frame-by-frame data if not requested
        if not include_frames:
            results.pop('frame_by_frame', None)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Analysis complete in {processing_time:.2f}s")
        
        return {
            "success": True,
            "filename": file.filename,
            "processing_time_seconds": round(processing_time, 2),
            "analysis": results
        }
        
    except Exception as e:
        logger.error(f"Error processing video: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
        
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# @app.post("/analyze-with-transcription")
# async def analyze_video_with_transcription(
#     file: UploadFile = File(...),
#     sample_rate: int = 1,
#     include_frames: bool = False,
#     transcribe: bool = True
# ):
#     """
#     Analyze emotions in uploaded video and transcribe audio
    
#     Args:
#         file: MP4 video file
#         sample_rate: Process every Nth frame (default: 1)
#         include_frames: Include frame-by-frame data in response (default: False)
#         transcribe: Enable audio transcription (default: True)
        
#     Returns:
#         Comprehensive emotion analysis with transcription
#     """
#     start_time = datetime.now()
    
#     logger.info(f"Received video + transcription request: {file.filename}")
    
#     # Validate file type
#     if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
#         raise HTTPException(
#             status_code=400,
#             detail="Invalid file format. Please upload MP4, AVI, MOV, or MKV video."
#         )
    
#     # Save uploaded file temporarily in the same directory as api_services.py
#     video_tmp_path = os.path.join(SCRIPT_DIR, f"_tmp_video_{os.getpid()}.mp4")
#     with open(video_tmp_path, 'wb') as tmp_video:
#         content = await file.read()
#         tmp_video.write(content)
#     video_path = video_tmp_path
    
#     audio_path = None
#     transcription_result = None
    
#     try:
#         # Analyze video emotions
#         logger.info("Starting emotion analysis...")
#         emotion_results = analyze_video_emotions(video_path, sample_rate=sample_rate)
        
#         # Extract and transcribe audio if requested
#         if transcribe:
#             audio_path = os.path.join(SCRIPT_DIR, f"_tmp_audio_{os.getpid()}.wav")
            
#             logger.info("Extracting audio from video...")
#             if extract_audio_from_video(video_path, audio_path):
#                 logger.info("Starting audio transcription...")
#                 transcription_result = transcribe_audio(audio_path)
#             else:
#                 logger.warning("Could not extract audio from video")
#                 transcription_result = {
#                     'success': False,
#                     'error': 'No audio stream found in video or extraction failed',
#                     'full_text': '',
#                     'segments': [],
#                     'segment_count': 0
#                 }
        
#         # Remove frame-by-frame data if not requested
#         if not include_frames:
#             emotion_results.pop('frame_by_frame', None)
        
#         processing_time = (datetime.now() - start_time).total_seconds()
        
#         logger.info(f"Complete analysis finished in {processing_time:.2f}s")
        
#         # Combine results
#         response = {
#             "success": True,
#             "filename": file.filename,
#             "processing_time_seconds": round(processing_time, 2),
#             "emotion_analysis": emotion_results,
#             "transcription": transcription_result if transcribe else None
#         }
        
#         return response
        
#     except Exception as e:
#         logger.error(f"Error processing video: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
        
#     finally:
#         # Clean up temporary files
#         if os.path.exists(video_path):
#             os.unlink(video_path)
#         if audio_path and os.path.exists(audio_path):
#             os.unlink(audio_path)

def cleanup_files(video_path, audio_path):
    import time
    for path in [video_path, audio_path]:
        for _ in range(3):
            try:
                if os.path.exists(path):
                    os.unlink(path)
                    logger.info(f"✓ Deleted temp file: {path}")
                    break
            except Exception as e:
                time.sleep(0.5)
        else:
            logger.warning(f"Could not delete temp file: {path}")


@app.post("/analyze-with-transcription-new")
async def analyze_video_with_transcription(
    file: UploadFile = File(...),
    sample_rate: int = 1,
    include_frames: bool = False,
    transcribe: bool = True,
    candidate_name: str = Query(default="Unknown", description="Candidate name"),
    role_applied: str = Query(default="", description="Job role being applied for")
):
    start_time = datetime.now()
    
    # --- video & audio temp file paths ---
    video_tmp_path = os.path.join(SCRIPT_DIR, f"_tmp_video_{uuid.uuid4()}.mp4")
    audio_path = os.path.join(SCRIPT_DIR, f"_tmp_audio_{uuid.uuid4()}.wav")
    
    try:
        # Save video temporarily
        with open(video_tmp_path, 'wb') as f:
            f.write(await file.read())
        
        logger.info(f"Video saved temporarily: {video_tmp_path}")
        
        # Analyze video emotions
        emotion_results = analyze_video_emotions(video_tmp_path, sample_rate=sample_rate)
        logger.info(f"emotion_results keys: {emotion_results.keys()}")
        logger.info(f"emotion_results['gaze_analysis']: {emotion_results.get('gaze_analysis', 'KEY NOT FOUND!')}")
        
        # --- audio extraction & analysis ---
        transcription_result = None
        nlp_result = None
        acoustic_result = None
        gaze_result = None
        
        if transcribe:
            audio_extracted = extract_audio_from_video(video_tmp_path, audio_path)
            if audio_extracted:
                logger.info(f"Audio extracted temporarily: {audio_path}")
                transcription_result = transcribe_audio(audio_path)
                speech_detected = bool(transcription_result.get("speech_detected", False))

                if speech_detected:
                    # Acoustic analysis on speech-positive clips only.
                    from utils.acoustic_analyzer import analyze_acoustics
                    acoustic_result = analyze_acoustics(audio_path)
                    acoustic_result["available"] = not bool(acoustic_result.get("error"))
                    acoustic_result["speech_detected"] = True
                    logger.info("✓ Acoustic analysis complete")

                    # NLP on transcript only when speech quality is reliable.
                    from utils.nlp_analyzer import analyze_transcript
                    nlp_result = analyze_transcript(transcription_result.get("full_text", ""))
                    nlp_result["available"] = not bool(nlp_result.get("error"))
                    logger.info("✓ NLP analysis complete")
                else:
                    acoustic_result = {
                        "available": False,
                        "speech_detected": False,
                        "reason": transcription_result.get("speech_reason", "No reliable speech detected"),
                    }
                    nlp_result = {
                        "available": False,
                        "error": "No reliable speech detected for NLP analysis",
                        "reason": transcription_result.get("speech_reason", "No reliable speech detected"),
                    }
                    logger.warning("Skipping acoustic/NLP due to non-speech or low-confidence transcription")
            else:
                transcription_result = {
                    "success": False,
                    "error": "No audio stream found in video or extraction failed",
                    "full_text": "",
                    "language": "unknown",
                    "segments": [],
                    "segment_count": 0,
                    "word_count": 0,
                    "avg_segment_confidence": 0.0,
                    "speech_detected": False,
                    "speech_reason": "Audio extraction failed",
                }
                acoustic_result = {
                    "available": False,
                    "speech_detected": False,
                    "reason": "Audio extraction failed",
                }
                nlp_result = {
                    "available": False,
                    "error": "Audio extraction failed; NLP not run",
                    "reason": "Audio extraction failed",
                }
        
        # Gaze summary from frame-by-frame (computed during emotion analysis)
        gaze_result = emotion_results.pop("gaze_analysis", None)
        logger.info(f"✓ Popped gaze_result: {gaze_result}")
        
        # Candidate composite score
        from utils.candidate_scorer import compute_candidate_score
        candidate_score = compute_candidate_score(
            emotion_results["emotion_analysis"],
            gaze_result,
            acoustic_result,
            nlp_result
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Analysis completed in {processing_time:.2f}s")
        logger.info(f"Final gaze_result being returned: {gaze_result}")
        
        return {
            "success": True,
            "candidate": {
                "name": candidate_name,
                "role_applied": role_applied,
                "interview_file": file.filename,
            },
            "candidate_score": candidate_score,
            "emotion_analysis": emotion_results,
            "gaze_analysis": gaze_result,
            "acoustic_analysis": acoustic_result,
            "verbal_analysis": {
                "transcription": transcription_result,
                "nlp": nlp_result
            },
            "processing_time_seconds": round(processing_time, 2)
        }
    
    except Exception as e:
        logger.error(f"Error processing video: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    
    finally: 
        logger.info("🔥 FINALLY BLOCK EXECUTING")
       # Clean up temp files
        cleanup_files(video_tmp_path, audio_path)



@app.post("/batch-analyze")
async def batch_analyze_videos(
    files: List[UploadFile] = File(...),
    sample_rate: int = 2
):
    """
    Analyze multiple videos in batch
    
    Args:
        files: List of video files
        sample_rate: Process every Nth frame (default: 2)
        
    Returns:
        List of analysis results
    """
    logger.info(f"Batch analysis request: {len(files)} videos")
    
    results = []
    
    for file in files:
        tmp_path = None
        try:
            # Process each video
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_path = tmp_file.name
            
            logger.info(f"Processing: {file.filename} (temp: {tmp_path})")
            analysis = analyze_video_emotions(tmp_path, sample_rate=sample_rate)
            
            results.append({
                "filename": file.filename,
                "success": True,
                "analysis": analysis
            })
            
            logger.info(f"✓ Completed: {file.filename}")
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}", exc_info=True)
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
        
        finally:
            # Ensure temp file is cleaned up
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                    logger.info(f"✓ Deleted temp file: {tmp_path}")
                except Exception as e:
                    logger.warning(f"Could not delete temp file {tmp_path}: {e}")
    
    logger.info(f"Batch analysis complete: {sum(1 for r in results if r['success'])}/{len(files)} successful")
    
    return {
        "batch_results": results,
        "total_videos": len(files),
        "successful": sum(1 for r in results if r['success'])
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)