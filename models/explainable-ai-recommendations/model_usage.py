"""
Enhanced interface with humanized feedback for job matching model
"""

import joblib
import pandas as pd
import numpy as np
import sys
from typing import Dict, List, Optional
from datetime import datetime

# Import semantic skill matcher with fallback
try:
    from semantic_skill_matcher import SkillMatcher, PROFICIENCY_MAP, proficiency_multiplier
    SEMANTIC_MATCHER_AVAILABLE = True
except ImportError as e:
    print(f"[Warning] Semantic skill matcher not available: {e}")
    SEMANTIC_MATCHER_AVAILABLE = False

# Import the model class and feedback generator
from model import (EnhancedMLModelTrainer, EnhancedFeatureEngineering,
                   EnhancedMLModelTrainerWithFeedback, HumanizedFeedbackGenerator)


class JobMatcherWithFeedback:
    """
    Enhanced wrapper with humanized feedback capabilities
    """

    def __init__(self, model_path: str, enable_feedback: bool = True):
        """
        Load the trained model with optional feedback generation

        Args:
            model_path: Path to the saved model file (.pkl)
            enable_feedback: Whether to enable humanized feedback (default: True)
        """
        print(f"📦 Loading model from {model_path}...")
        
        # Fix for pickle loading issue: map __main__ module classes to model module
        # This handles models saved when running scripts directly (they reference __main__)
        # Ensure __main__ exists
        if '__main__' not in sys.modules:
            import types
            sys.modules['__main__'] = types.ModuleType('__main__')
        
        main_module = sys.modules['__main__']
        had_classes = hasattr(main_module, 'EnhancedMLModelTrainer')
        
        # Temporarily add required classes to __main__ module for pickle to find them
        if not had_classes:
            main_module.EnhancedMLModelTrainer = EnhancedMLModelTrainer
            main_module.EnhancedFeatureEngineering = EnhancedFeatureEngineering
            main_module.EnhancedMLModelTrainerWithFeedback = EnhancedMLModelTrainerWithFeedback
            main_module.HumanizedFeedbackGenerator = HumanizedFeedbackGenerator
        
        try:
            self.model = joblib.load(model_path)
        finally:
            # Clean up: remove the patched classes if we added them
            if not had_classes:
                if hasattr(main_module, 'EnhancedMLModelTrainer'):
                    delattr(main_module, 'EnhancedMLModelTrainer')
                if hasattr(main_module, 'EnhancedFeatureEngineering'):
                    delattr(main_module, 'EnhancedFeatureEngineering')
                if hasattr(main_module, 'EnhancedMLModelTrainerWithFeedback'):
                    delattr(main_module, 'EnhancedMLModelTrainerWithFeedback')
                if hasattr(main_module, 'HumanizedFeedbackGenerator'):
                    delattr(main_module, 'HumanizedFeedbackGenerator')

        if not self.model.is_trained:
            raise ValueError("Loaded model is not trained!")

        # Initialize feedback generator if enabled
        self.enable_feedback = enable_feedback
        if self.enable_feedback:
            # Check if model already has feedback generator
            if not hasattr(self.model, 'feedback_generator') or self.model.feedback_generator is None:
                self.model.feedback_generator = HumanizedFeedbackGenerator(self.model)
                print("✅ Feedback generator initialized")

        print(f"✅ Model loaded successfully!")
        print(f"   Model type: {self.model.model_type}")
        print(f"   Features used: {len(self.model.feature_names)}")
        print(f"   Feedback enabled: {self.enable_feedback}")

    def match_student_with_job(self, student: Dict, job: Dict,
                               feedback_mode: str = 'full') -> Dict:
        """
        Match a student with a job and return prediction with humanized feedback

        Args:
            student: Student data dictionary
            job: Job data dictionary
            feedback_mode: Type of feedback to generate
                - 'full': Complete narrative feedback with all sections
                - 'quick': Summary feedback with key insights only
                - 'structured': Organized data without narrative
                - 'none': Just scores, no feedback

        Returns:
            Dictionary with prediction results and humanized feedback
        """
        print("\n🔍 Analyzing match...")

        # Step 1: Extract features from student and job
        features = self._extract_features(student, job)

        # Step 2: Add advanced features
        features = self._add_advanced_features(features)

        # Step 3: Make prediction
        predicted_score = self.model.predict(features)

        # Step 4: Classify into category
        category = self._classify_score(predicted_score)

        # Step 5: Prepare candidate profile for feedback
        candidate_profile = self._prepare_candidate_profile(student, features)
        job_requirements = self._prepare_job_requirements(job)

        # Step 6: Generate feedback based on mode
        feedback_data = None
        if self.enable_feedback and feedback_mode != 'none':
            feedback_data = self._generate_feedback(
                candidate_profile,
                job_requirements,
                predicted_score,
                features,
                feedback_mode
            )

        # Build result dictionary
        result = {
            'match_score': round(predicted_score, 2),
            'recommendation_type': category,
            'component_scores': {
                'skills': features.get('skills_score', 0),
                'education': features.get('education_score', 0),
                'projects': features.get('projects_score', 0),
                'courses': features.get('courses_score', 0),
                'cgpa': features.get('cgpa_score', 0)
            },
            'basic_explanation': self._generate_basic_explanation(features, predicted_score, category)
        }

        # Add feedback if generated
        if feedback_data:
            result['humanized_feedback'] = feedback_data

        # Print results
        self._print_results(result, feedback_mode)

        return result

    def _prepare_candidate_profile(self, student: Dict, features: Dict) -> Dict:
        """Prepare candidate profile for feedback generation"""
        profile = {
            'name': student.get('name', 'Candidate'),
            'cgpa': student.get('cgpa', 0),
            'degree': student.get('dept_name', 'degree'),
            'university': student.get('university', 'your institution'),
            'batch': student.get('batch', datetime.now().year),
            'experience_years': self._calculate_experience(student),

            # Skills data
            'skills': [s['skill_name'] for s in student.get('skills', [])],
            'matched_skills': self._get_matched_skills(student, features),
            'matched_skills_count': int(features.get('matched_skills_count', 0)),
            'missing_skills_count': int(features.get('missing_skills_count', 0)),

            # Component scores
            'skills_score': features.get('skills_score', 0),
            'education_score': features.get('education_score', 0),
            'projects_score': features.get('projects_score', 0),
            'courses_score': features.get('courses_score', 0),
            'cgpa_score': features.get('cgpa_score', 0),

            # Additional data
            'projects_count': len(student.get('projects', [])),
            'project_domains': list(set([p.get('domain', 'various') for p in student.get('projects', [])])),
            'courses_count': len(student.get('courses', [])),
            'cgpa_excess': features.get('cgpa_excess', 0),
            'mandatory_missing': int(features.get('mandatory_missing', 0))
        }

        return profile

    def _prepare_job_requirements(self, job: Dict) -> Dict:
        """Prepare job requirements for feedback generation"""
        requirements = {
            'title': job.get('title', 'this position'),
            'description': job.get('description', ''),
            'required_skills': [s['skill_name'] for s in job.get('required_skills', [])],
            'mandatory_skills': [s['skill_name'] for s in job.get('required_skills', [])
                               if s.get('is_mandatory', False)],
            'min_cgpa': job.get('eligible_cgpa_min', 0),
            'min_experience_years': job.get('min_experience_years', 0),
            'category': self._infer_job_category(job),
            'company': job.get('company', 'the organization')
        }

        return requirements

    def _calculate_experience(self, student: Dict) -> float:
        """Calculate approximate years of experience"""
        current_year = datetime.now().year
        batch_year = student.get('batch', current_year)

        # Estimate based on projects, internships, etc.
        projects = len(student.get('projects', []))
        internships = len(student.get('internships', []))

        years = max(0, current_year - batch_year - 2)  # Years after expected graduation
        years += (projects * 0.2)  # Each project adds ~2-3 months equivalent
        years += (internships * 0.5)  # Each internship adds ~6 months

        return round(years, 1)

    def _get_matched_skills(self, student: Dict, features: Dict) -> List[str]:
        """Extract list of matched skills"""
        # This is a simplified version - enhance based on your feature extraction
        return [s['skill_name'] for s in student.get('skills', [])][:int(features.get('matched_skills_count', 0))]

    def _infer_job_category(self, job: Dict) -> str:
        """Infer job category from title/description"""
        title_lower = job.get('title', '').lower()
        desc_lower = job.get('description', '').lower()

        categories = {
            'data_science': ['data scientist', 'machine learning', 'ml engineer', 'ai'],
            'web_development': ['web developer', 'frontend', 'backend', 'full stack', 'react', 'node'],
            'cybersecurity': ['security', 'cybersecurity', 'penetration', 'ethical hacker'],
            'mobile_dev': ['mobile', 'android', 'ios', 'flutter', 'react native'],
            'devops': ['devops', 'sre', 'infrastructure', 'kubernetes', 'docker']
        }

        for category, keywords in categories.items():
            if any(kw in title_lower or kw in desc_lower for kw in keywords):
                return category

        return 'general'

    def _generate_feedback(self, candidate_profile: Dict, job_requirements: Dict,
                          match_score: float, features: Dict, mode: str) -> Dict:
        """Generate humanized feedback based on mode"""

        if not self.model.feedback_generator:
            return {'error': 'Feedback generator not available'}

        # Get feature importance (if available)
        feature_importance = {}
        if hasattr(self.model.model, 'feature_importances_'):
            feature_importance = dict(zip(self.model.feature_names,
                                        self.model.model.feature_importances_))

        if mode == 'full':
            # Generate complete narrative feedback
            narrative = self.model.feedback_generator.generate_comprehensive_feedback(
                candidate_profile, job_requirements, match_score, feature_importance
            )

            return {
                'type': 'full_narrative',
                'content': narrative,
                'sections': self._parse_feedback_sections(narrative)
            }

        elif mode == 'quick':
            # Generate condensed feedback
            return self._generate_quick_feedback(
                candidate_profile, job_requirements, match_score
            )

        elif mode == 'structured':
            # Generate structured data feedback
            return {
                'type': 'structured',
                'strengths': self.model.feedback_generator._analyze_strengths(
                    candidate_profile, feature_importance
                ),
                'gaps': self.model.feedback_generator._detailed_gap_analysis(
                    candidate_profile, job_requirements
                ),
                'action_plan': self.model.feedback_generator._generate_action_plan(
                    candidate_profile, job_requirements
                ),
                'timeline': self.model.feedback_generator._estimate_readiness_timeline(
                    candidate_profile, job_requirements
                ),
                'encouragement': self.model.feedback_generator._personalized_encouragement(
                    match_score, candidate_profile
                )
            }

        return {}

    def _generate_quick_feedback(self, profile: Dict, requirements: Dict,
                                score: float) -> Dict:
        """Generate condensed quick feedback"""

        if score >= 85:
            headline = f"🎉 Excellent match, {profile['name']}! You're ready to apply."
        elif score >= 70:
            headline = f"👍 Strong match, {profile['name']}! Minor improvements needed."
        elif score >= 50:
            headline = f"💡 Potential fit, {profile['name']}. Some skill development required."
        else:
            headline = f"📚 Significant upskilling needed, {profile['name']}."

        # Quick stats
        matched = profile.get('matched_skills_count', 0)
        missing = profile.get('missing_skills_count', 0)
        mandatory_missing = profile.get('mandatory_missing', 0)

        quick_summary = []
        quick_summary.append(f"✅ {matched} skills matched")
        if missing > 0:
            quick_summary.append(f"📝 {missing} skills to develop")
        if mandatory_missing > 0:
            quick_summary.append(f"⚠️ {mandatory_missing} critical skills missing")

        # Top 2 recommendations
        top_actions = []
        if missing > 0:
            top_actions.append("Enroll in courses for missing skills")
        if profile.get('projects_count', 0) < 3:
            top_actions.append("Build 1-2 portfolio projects")
        if not top_actions:
            top_actions.append("Update resume and apply now!")

        return {
            'type': 'quick_summary',
            'headline': headline,
            'summary_points': quick_summary,
            'top_2_actions': top_actions[:2],
            'estimated_timeline': self._quick_timeline_estimate(score, missing)
        }

    def _quick_timeline_estimate(self, score: float, missing_count: int) -> str:
        """Quick timeline estimate"""
        if score >= 85:
            return "Ready now"
        elif missing_count <= 2:
            return "1-2 months"
        elif missing_count <= 5:
            return "3-4 months"
        else:
            return "6+ months"

    def _parse_feedback_sections(self, narrative: str) -> Dict:
        """Parse narrative into sections for easier access"""
        sections = {}
        current_section = None
        current_content = []

        for line in narrative.split('\n'):
            if line.startswith('##'):
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = line.replace('##', '').strip()
                current_content = []
            else:
                current_content.append(line)

        if current_section:
            sections[current_section] = '\n'.join(current_content)

        return sections

    # [Keep all existing helper methods from SimpleJobMatcher]
    def _extract_features(self, student: Dict, job: Dict) -> Dict:
        """Extract basic features from student and job data"""
        features = {}

        # 1. Skills Score
        student_skills = {s['skill_name']: s['proficiency_level'] for s in student.get('skills', [])}
        required_skills = job.get('required_skills', [])

        skills_result = self._score_skills(student_skills, required_skills)
        features['skills_score'] = skills_result['score']
        features['matched_skills_count'] = skills_result['matched_count']
        features['missing_skills_count'] = skills_result['missing_count']
        features['mandatory_missing'] = skills_result['mandatory_missing']

        # 2. Education Score
        education_result = self._score_education(student, job)
        features['education_score'] = education_result['score']

        # 3. Projects Score
        projects = student.get('projects', [])
        projects_result = self._score_projects(projects, required_skills)
        features['projects_score'] = projects_result['score']

        # 4. Courses Score
        courses = student.get('courses', [])
        courses_result = self._score_courses(courses, job)
        features['courses_score'] = courses_result['score']

        # 5. CGPA Score
        cgpa_result = self._score_cgpa(student, job)
        features['cgpa_score'] = cgpa_result['score']
        features['cgpa_excess'] = cgpa_result['excess']

        # 6. Graduation year multiplier
        features['graduation_year_multiplier'] = self._calculate_grad_multiplier(student)

        # 7. Interaction features
        features['skills_x_education'] = (features['skills_score'] * features['education_score']) / 100
        features['projects_x_courses'] = (features['projects_score'] * features['courses_score']) / 100

        return features

    def _score_skills(self, student_skills: Dict, required_skills: List[Dict]) -> Dict:
        """Score skills match with semantic matching support"""
        if not required_skills:
            return {'score': 100, 'matched_count': 0, 'missing_count': 0, 'mandatory_missing': 0}

        matched_count = 0
        missing_count = 0
        mandatory_missing = 0
        total_weight = sum(s.get('weight', 1.0) for s in required_skills)
        matched_weight = 0

        # Initialize semantic matcher with student's skill names
        student_skill_names = list(student_skills.keys())
        print(f"\n[SkillMatching] Student skills: {student_skill_names}")
        print(f"[SkillMatching] Required skills count: {len(required_skills)}")
        
        matcher = None
        if SEMANTIC_MATCHER_AVAILABLE and student_skill_names:
            try:
                matcher = SkillMatcher(student_skill_names, threshold=0.25)
                print(f"[SkillMatching] ✓ Semantic matcher initialized with {len(student_skill_names)} student skills (threshold=0.25)")
            except Exception as e:
                print(f"[SkillMatching] ✗ Failed to initialize semantic matcher: {e}")
                matcher = None
        else:
            print(f"[SkillMatching] Matcher not available. SEMANTIC_MATCHER_AVAILABLE={SEMANTIC_MATCHER_AVAILABLE}, skills_count={len(student_skill_names)}")

        for idx, req_skill in enumerate(required_skills, 1):
            req_skill_name = req_skill['skill_name']
            required_level = req_skill.get('required_level', 'Intermediate')
            is_mandatory = req_skill.get('is_mandatory', False)
            weight = req_skill.get('weight', 1.0)

            # Try exact match first, then semantic match
            matched_skill_name = None
            similarity_score = 1.0
            match_type = 'none'
            
            if req_skill_name in student_skills:
                matched_skill_name = req_skill_name
                match_type = 'exact'
                print(f"  [{idx:2d}] '{req_skill_name}' → EXACT MATCH")
            elif matcher:
                # Try semantic match if exact match fails
                try:
                    matched_skill_name, similarity_score = matcher.best_match(req_skill_name)
                    if matched_skill_name:
                        match_type = f'semantic({similarity_score:.3f})'
                        print(f"  [{idx:2d}] '{req_skill_name}' → '{matched_skill_name}' (sim: {similarity_score:.3f}) ✓")
                    else:
                        print(f"  [{idx:2d}] '{req_skill_name}' → best: {similarity_score:.3f} (below 0.25 threshold)")
                except Exception as e:
                    print(f"  [{idx:2d}] '{req_skill_name}' → ERROR: {e}")
                    matched_skill_name = None
            else:
                print(f"  [{idx:2d}] '{req_skill_name}' → MATCHER NOT AVAILABLE")

            if matched_skill_name:
                student_level = student_skills[matched_skill_name]

                # Get proficiency levels
                candidate_prof_level = PROFICIENCY_MAP.get(student_level, 1)
                required_prof_level = PROFICIENCY_MAP.get(required_level, 1)

                # Calculate proficiency match with similarity boost
                proficiency_mult = proficiency_multiplier(candidate_prof_level, required_prof_level)
                proficiency_match = min(1.0, proficiency_mult * similarity_score)
                print(f"       cand_level={student_level}({candidate_prof_level}) vs req_level={required_level}({required_prof_level}) → {proficiency_match:.3f}")

                matched_weight += weight * proficiency_match
                matched_count += 1
            else:
                missing_count += 1
                if is_mandatory:
                    mandatory_missing += 1

        score = (matched_weight / total_weight * 100) if total_weight > 0 else 0

        return {
            'score': round(score, 2),
            'matched_count': matched_count,
            'missing_count': missing_count,
            'mandatory_missing': mandatory_missing
        }

    def _score_education(self, student: Dict, job: Dict) -> Dict:
        """Score education eligibility"""
        score = 0

        student_batch = student.get('batch', 0)
        eligible_batches = job.get('eligible_batches', [])
        if student_batch in eligible_batches:
            score += 40

        student_dept = student.get('dept_name', '')
        eligible_depts = job.get('eligible_departments', [])
        if any(dept in student_dept for dept in eligible_depts):
            score += 40

        student_cgpa = student.get('cgpa', 0)
        min_cgpa = job.get('eligible_cgpa_min', 0)
        if student_cgpa >= min_cgpa:
            score += 20

        return {'score': score}

    def _score_projects(self, projects: List[Dict], required_skills: List[Dict]) -> Dict:
        """Score relevant projects"""
        if not projects:
            return {'score': 0}

        required_skill_names = set(s['skill_name'] for s in required_skills)
        relevant_count = 0

        for project in projects:
            project_skills = set(project.get('skills', []))
            if project_skills.intersection(required_skill_names):
                relevant_count += 1

        score = min(relevant_count * 25, 100)
        return {'score': score}

    def _score_courses(self, courses: List[Dict], job: Dict) -> Dict:
        """Score relevant coursework"""
        if not courses:
            return {'score': 0}

        job_keywords = set(job.get('title', '').lower().split() +
                          job.get('description', '').lower().split())

        relevant_courses = []
        for course in courses:
            course_name = course.get('course_name', '').lower()
            if any(keyword in course_name for keyword in job_keywords if len(keyword) > 3):
                relevant_courses.append(course)

        if not relevant_courses:
            return {'score': 0}

        avg_grade = np.mean([c.get('gpa', 0) for c in relevant_courses])
        score = min(len(relevant_courses) * 20 + avg_grade * 20, 100)

        return {'score': round(score, 2)}

    def _score_cgpa(self, student: Dict, job: Dict) -> Dict:
        """Score CGPA performance"""
        student_cgpa = student.get('cgpa', 0)
        min_cgpa = job.get('eligible_cgpa_min', 0)

        if student_cgpa < min_cgpa:
            score = 0
        else:
            score = min((student_cgpa / 4.0) * 100, 100)

        return {
            'score': round(score, 2),
            'excess': student_cgpa - min_cgpa
        }

    def _calculate_grad_multiplier(self, student: Dict) -> float:
        """Calculate graduation year multiplier"""
        current_year = datetime.now().year
        batch_year = student.get('batch', current_year)
        years_in_program = current_year - batch_year

        if years_in_program >= 4:
            return 1.15
        elif years_in_program == 3:
            return 1.10
        elif years_in_program == 2:
            return 1.05
        else:
            return 1.00

    def _add_advanced_features(self, basic_features: Dict) -> Dict:
        """Add advanced engineered features"""
        temp_df = pd.DataFrame([basic_features])
        temp_df = EnhancedFeatureEngineering.create_advanced_features(temp_df)
        return temp_df.iloc[0].to_dict()

    def _classify_score(self, score: float) -> str:
        """Classify score into recommendation category"""
        if score >= 85:
            return 'Perfect_Match'
        elif score >= 70:
            return 'Good_Match'
        elif score >= 50:
            return 'Potential_Match'
        else:
            return 'Upskill_Opportunity'

    def _generate_basic_explanation(self, features: Dict, score: float, category: str) -> str:
        """Generate human-readable basic explanation"""
        explanations = []

        if category == 'Perfect_Match':
            explanations.append("🌟 Excellent match! This candidate meets all requirements.")
        elif category == 'Good_Match':
            explanations.append("👍 Strong match with minor gaps that can be addressed.")
        elif category == 'Potential_Match':
            explanations.append("💡 Potential candidate with some development needed.")
        else:
            explanations.append("📚 Significant upskilling required for this role.")

        components = [
            ('Skills', features.get('skills_score', 0)),
            ('Education', features.get('education_score', 0)),
            ('Projects', features.get('projects_score', 0)),
            ('Courses', features.get('courses_score', 0)),
            ('CGPA', features.get('cgpa_score', 0))
        ]

        strengths = [name for name, score in components if score >= 80]
        weaknesses = [name for name, score in components if score < 50]

        if strengths:
            explanations.append(f"Strengths: {', '.join(strengths)}")

        if weaknesses:
            explanations.append(f"Areas for improvement: {', '.join(weaknesses)}")

        if features.get('mandatory_missing', 0) > 0:
            explanations.append(f"⚠️ Missing {features['mandatory_missing']} mandatory skill(s)")

        return " | ".join(explanations)

    def _print_results(self, result: Dict, feedback_mode: str):
        """Print results in a nice format with feedback"""
        print("\n" + "="*60)
        print("📊 MATCH RESULTS")
        print("="*60)

        print(f"\n🎯 Overall Match Score: {result['match_score']:.2f}/100")
        print(f"📋 Recommendation: {result['recommendation_type']}")

        print(f"\n📈 Component Breakdown:")
        for component, score in result['component_scores'].items():
            bar = "█" * int(score / 5)
            print(f"   {component.capitalize():12} [{score:5.1f}] {bar}")

        print(f"\n💬 {result['basic_explanation']}")

        # Print humanized feedback based on mode
        if 'humanized_feedback' in result:
            feedback = result['humanized_feedback']

            if feedback_mode == 'full':
                print("\n" + "="*60)
                print("📝 HUMANIZED FEEDBACK")
                print("="*60)
                print(feedback['content'])

            elif feedback_mode == 'quick':
                print("\n" + "="*60)
                print("⚡ QUICK FEEDBACK")
                print("="*60)
                print(f"\n{feedback['headline']}\n")
                for point in feedback['summary_points']:
                    print(f"  {point}")
                print(f"\n🎯 Next Steps:")
                for i, action in enumerate(feedback['top_2_actions'], 1):
                    print(f"  {i}. {action}")
                print(f"\n⏰ Timeline: {feedback['estimated_timeline']}")

            elif feedback_mode == 'structured':
                print("\n" + "="*60)
                print("📊 STRUCTURED INSIGHTS")
                print("="*60)

                print(f"\n🌟 Top Strengths:")
                for strength in feedback.get('strengths', [])[:3]:
                    print(f"  • {strength['category']}: {strength['message'][:80]}...")

                print(f"\n🎯 Critical Gaps:")
                for gap in feedback.get('gaps', []):
                    if gap.get('priority') == 'critical':
                        print(f"  🔴 {gap['category']}: {len(gap.get('missing', []))} items")

                print(f"\n⏰ Timeline: {feedback.get('timeline', {}).get('message', 'N/A')}")

        print("="*60)

# ============================================================================
# DEMO USAGE WITH DIFFERENT FEEDBACK MODES
# ============================================================================

def demo_with_feedback():
    """
    Demo showing different feedback modes
    """

    # Load the model with feedback enabled
    matcher = JobMatcherWithFeedback(
        'xai_gradient_boosting.pkl',
        enable_feedback=True
    )

    # Define a student
    student = {
        's_id': 'demo_001',
        'name': 'Alice Johnson',
        'cgpa': 3.7,
        'batch': 2022,
        'dept_name': 'Computer Science',
        'university': 'State University',
        'skills': [
            {'skill_name': 'Python', 'proficiency_level': 'Advanced'},
            {'skill_name': 'Machine Learning', 'proficiency_level': 'Intermediate'},
            {'skill_name': 'SQL', 'proficiency_level': 'Intermediate'},
        ],
        'projects': [
            {
                'title': 'ML Predictor',
                'skills': ['Python', 'Machine Learning', 'Pandas'],
                'domain': 'Machine Learning'
            },
        ],
        'courses': [
            {'course_name': 'Artificial Intelligence', 'gpa': 3.8},
            {'course_name': 'Database Systems', 'gpa': 3.6},
        ]
    }

    # Define a job
    job = {
        'j_id': 'job_001',
        'title': 'Data Scientist',
        'company': 'Tech Corp',
        'description': 'Looking for a data scientist with Python and ML skills',
        'required_skills': [
            {'skill_name': 'Python', 'required_level': 'Advanced', 'is_mandatory': True, 'weight': 2.0},
            {'skill_name': 'Machine Learning', 'required_level': 'Intermediate', 'is_mandatory': True, 'weight': 1.5},
            {'skill_name': 'Deep Learning', 'required_level': 'Intermediate', 'is_mandatory': False, 'weight': 1.0},
            {'skill_name': 'TensorFlow', 'required_level': 'Intermediate', 'is_mandatory': False, 'weight': 1.0},
        ],
        'eligible_batches': [2021, 2022, 2023],
        'eligible_departments': ['Computer Science', 'Data Science'],
        'eligible_cgpa_min': 3.0,
        'min_experience_years': 0
    }

    print("\n" + "="*70)
    print("DEMO 1: FULL NARRATIVE FEEDBACK")
    print("="*70)
    result_full = matcher.match_student_with_job(student, job, feedback_mode='full')

    print("\n" + "="*70)
    print("DEMO 2: QUICK FEEDBACK")
    print("="*70)
    result_quick = matcher.match_student_with_job(student, job, feedback_mode='quick')

    print("\n" + "="*70)
    print("DEMO 3: STRUCTURED FEEDBACK")
    print("="*70)
    result_structured = matcher.match_student_with_job(student, job, feedback_mode='structured')

    return result_full


if __name__ == "__main__":
    demo_with_feedback()
