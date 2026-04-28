"""
semantic_skill_matcher.py
==========================
Dynamic skill matching using all-MiniLM-L6-v2 sentence embeddings.

Key improvements over v1
-------------------------
1. Context augmentation — each skill name is expanded with a short semantic
   descriptor before embedding. This bridges the gap between abstract job
   requirements ("browser automation", "LLM integration") and concrete tool
   names ("Selenium", "OpenAI API") that the model would otherwise score low.

   Example:
     "Selenium"   → "Selenium browser automation end-to-end testing tool"
     "OpenAI API" → "OpenAI API LLM language model AI integration GPT"
     "Node.js"    → "Node.js server-side JavaScript runtime backend"

2. Two-pass matching:
     Pass 1 — embed both sides with context, cosine similarity
     Pass 2 — embed both sides as raw cleaned names
   Final score = max(pass1, pass2) — catches both exact variants and concepts.

3. Threshold lowered to 0.40 (was 0.60).
   Context augmentation keeps false positives low even at this threshold.

4. Shared model instance — loaded once per process, never reloaded.

Install:
    pip install sentence-transformers scikit-learn numpy

Model: all-MiniLM-L6-v2 (~80 MB, downloads once and is cached)
Set MINILM_MODEL_PATH env var to use a local copy:
    export MINILM_MODEL_PATH=/path/to/all-MiniLM-L6-v2
"""

from __future__ import annotations

import os
import re
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Shared model singleton
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None
_DEFAULT_MODEL_DIRS = [
    Path(__file__).resolve().parents[1] / _MODEL_NAME,
    Path(__file__).resolve().parents[1] / _MODEL_NAME,
]


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            env_path = os.environ.get("MINILM_MODEL_PATH", "").strip()

            if env_path:
                model_dir = Path(env_path)
            else:
                model_dir = next((p for p in _DEFAULT_MODEL_DIRS if p.exists()), None)
                if model_dir is None:
                    # Fall back to auto-download from HuggingFace
                    print(f"[SkillMatcher] Local model not found, downloading {_MODEL_NAME} from HuggingFace...")
                    _model = SentenceTransformer(_MODEL_NAME)
                    print("[SkillMatcher] Model ready.")
                    return _model

            print(f"[SkillMatcher] Loading {_MODEL_NAME} from local path: {model_dir}")
            _model = SentenceTransformer(str(model_dir), local_files_only=True)
            print("[SkillMatcher] Model ready.")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required.\n"
                "Install:  pip install sentence-transformers"
            )
    return _model


# ─────────────────────────────────────────────────────────────────────────────
# Context augmentation map
# ─────────────────────────────────────────────────────────────────────────────
# Maps lower-cased substrings → semantic context tokens.
# Unknown skills embed as-is (still handles exact/near-exact names well).
# Add entries freely — keys are substrings, so "selenium" matches
# "Selenium", "selenium-webdriver", "selenium grid", etc.

CONTEXT_MAP: Dict[str, str] = {
    # ── Browser / test automation ────────────────────────────────────────
    "selenium":       "Selenium browser automation end-to-end testing web scraping",
    "playwright":     "Playwright browser automation end-to-end testing web scraping",
    "puppeteer":      "Puppeteer browser automation headless chrome web scraping",
    "cypress":        "Cypress browser automation end-to-end frontend testing",
    "jest":           "Jest JavaScript unit testing automated testing framework",
    "pytest":         "pytest Python unit testing automated testing framework",
    "mocha":          "Mocha JavaScript unit testing automated testing framework",

    # ── AI / LLM / ML ────────────────────────────────────────────────────
    "openai":         "OpenAI API GPT LLM language model AI integration chatbot",
    "claude":         "Claude Anthropic API LLM language model AI integration",
    "langchain":      "LangChain LLM language model AI pipeline integration framework",
    "n8n":            "n8n workflow automation low-code integration pipeline",
    "huggingface":    "HuggingFace transformers NLP machine learning model hub",
    "tensorflow":     "TensorFlow machine learning deep learning neural network",
    "pytorch":        "PyTorch machine learning deep learning neural network",
    "scikit":         "scikit-learn machine learning classification regression",
    "keras":          "Keras deep learning neural network model training",
    "nlp":            "NLP natural language processing text analysis pipeline",
    "regression":     "regression model machine learning prediction statistical",

    # ── Node / JS backend ────────────────────────────────────────────────
    "node":           "Node.js server-side JavaScript runtime backend API",
    "express":        "Express.js Node.js REST API backend web framework",
    "nestjs":         "NestJS Node.js backend framework TypeScript REST API",
    "fastify":        "Fastify Node.js backend REST API web framework",
    "deno":           "Deno JavaScript TypeScript server-side runtime backend",

    # ── Python backend ───────────────────────────────────────────────────
    "django":         "Django Python web framework backend REST API ORM",
    "fastapi":        "FastAPI Python REST API backend async framework",
    "flask":          "Flask Python web framework backend REST API lightweight",
    "celery":         "Celery Python task queue background jobs async worker",

    # ── Frontend ─────────────────────────────────────────────────────────
    "react":          "React.js frontend JavaScript UI library component",
    "next":           "Next.js React frontend SSR server-side rendering framework",
    "vue":            "Vue.js frontend JavaScript UI framework component",
    "angular":        "Angular frontend TypeScript framework SPA component",
    "svelte":         "Svelte frontend JavaScript UI framework reactive",
    "tailwind":       "Tailwind CSS utility-first styling frontend design",
    "framer":         "Framer Motion animation React frontend UI library",
    "redux":          "Redux state management React frontend JavaScript",
    "graphql":        "GraphQL API query language schema type-safe data fetching",

    # ── Databases ────────────────────────────────────────────────────────
    "mongodb":        "MongoDB NoSQL document database JSON storage",
    "postgres":       "PostgreSQL relational SQL database RDBMS",
    "mysql":          "MySQL relational SQL database RDBMS",
    "redis":          "Redis in-memory cache key-value database pub-sub",
    "elasticsearch":  "Elasticsearch search engine full-text indexing analytics",
    "cassandra":      "Cassandra distributed NoSQL wide-column database",
    "sqlite":         "SQLite embedded relational SQL database lightweight",
    "dynamodb":       "DynamoDB AWS NoSQL key-value document database serverless",
    "prisma":         "Prisma ORM database schema migration TypeScript",
    "sequelize":      "Sequelize ORM SQL database Node.js JavaScript",
    "mongoose":       "Mongoose MongoDB ORM Node.js JavaScript schema",

    # ── DevOps / Cloud ───────────────────────────────────────────────────
    "docker":         "Docker container containerization deployment microservices",
    "kubernetes":     "Kubernetes container orchestration deployment scaling k8s",
    "ci/cd":          "CI/CD continuous integration delivery pipeline GitHub Actions",
    "github actions": "GitHub Actions CI/CD continuous integration deployment pipeline",
    "terraform":      "Terraform infrastructure as code IaC cloud provisioning",
    "ansible":        "Ansible configuration management automation infrastructure",
    "aws":            "AWS Amazon Web Services cloud hosting S3 EC2 Lambda",
    "gcp":            "GCP Google Cloud Platform cloud hosting Kubernetes",
    "azure":          "Azure Microsoft Cloud hosting DevOps cloud services",
    "nginx":          "Nginx web server reverse proxy load balancer",
    "linux":          "Linux Unix server operating system bash shell",

    # ── Auth / Security ──────────────────────────────────────────────────
    "jwt":            "JWT JSON Web Token authentication authorization bearer",
    "oauth":          "OAuth2 authentication authorization third-party login SSO",
    "rbac":           "RBAC role-based access control permissions authorization",
    "ssl":            "SSL TLS HTTPS encryption secure certificate",

    # ── APIs / Protocols ─────────────────────────────────────────────────
    "rest":           "RESTful REST API HTTP backend endpoints JSON",
    "websocket":      "WebSocket real-time bidirectional communication socket",
    "grpc":           "gRPC remote procedure call protocol microservices",
    "mqtt":           "MQTT IoT messaging protocol lightweight pub-sub",
    "swagger":        "Swagger OpenAPI REST API documentation specification",

    # ── Design / CAD / Engineering ───────────────────────────────────────
    "autocad":        "AutoCAD CAD computer-aided design 2D 3D drafting engineering",
    "solidworks":     "SolidWorks 3D CAD mechanical design engineering modeling",
    "catia":          "CATIA 3D CAD design aerospace automotive engineering",
    "revit":          "Revit BIM building information modeling architecture design",
    "blender":        "Blender 3D modeling animation rendering design",
    "figma":          "Figma UI UX design prototyping wireframe collaboration",
    "sketch":         "Sketch UI UX design prototyping wireframe macOS",
    "matlab":         "MATLAB numerical computing matrix simulation engineering",
    "labview":        "LabVIEW National Instruments data acquisition measurement",

    # ── Mobile ───────────────────────────────────────────────────────────
    "flutter":        "Flutter Dart mobile app development iOS Android cross-platform",
    "react native":   "React Native mobile app development iOS Android JavaScript",
    "swift":          "Swift iOS macOS Apple mobile app development",
    "kotlin":         "Kotlin Android mobile app development JVM",
    "android":        "Android mobile app development Java Kotlin Google",
    "ios":            "iOS iPhone mobile app development Swift Objective-C",

    # ── Data / Analytics ─────────────────────────────────────────────────
    "pandas":         "pandas Python data analysis DataFrame tabular data",
    "numpy":          "NumPy Python numerical computing arrays matrix math",
    "spark":          "Apache Spark big data distributed processing analytics",
    "airflow":        "Apache Airflow workflow orchestration data pipeline ETL",
    "dbt":            "dbt data transformation SQL analytics engineering",
    "tableau":        "Tableau data visualization business intelligence dashboard",
    "powerbi":        "Power BI Microsoft data visualization business intelligence",

    # ── Messaging / Queue ────────────────────────────────────────────────
    "kafka":          "Apache Kafka event streaming message queue distributed",
    "rabbitmq":       "RabbitMQ message broker queue AMQP async communication",
    "sqs":            "AWS SQS Simple Queue Service message queue async",

    # ── Shopify / E-commerce ─────────────────────────────────────────────
    "shopify":        "Shopify e-commerce platform store merchant online shop",
    "liquid":         "Shopify Liquid template language theme customization",
    "woocommerce":    "WooCommerce WordPress e-commerce store plugin",

    # ── Notifications ────────────────────────────────────────────────────
    "fcm":            "FCM Firebase Cloud Messaging push notification mobile",
    "firebase":       "Firebase Google backend mobile real-time database push",
    "apns":           "APNs Apple Push Notification Service iOS push notification",

    # ── Web scraping / automation ─────────────────────────────────────────
    "beautifulsoup":  "BeautifulSoup Python web scraping HTML parsing",
    "scrapy":         "Scrapy Python web scraping crawling spider framework",
    "proxy":          "proxy IP rotation anonymization web scraping bypass",
    "captcha":        "CAPTCHA solving bypass automation anti-bot",

    # ── Additional Frontend/Web ──────────────────────────────────────────
    "html":           "HTML hypertext markup language web frontend structure",
    "css":            "CSS cascading stylesheets web frontend styling design",
    "typescript":     "TypeScript JavaScript type-safe programming language",
    "javascript":     "JavaScript programming language web frontend backend",
    "full-stack":     "full-stack web development frontend backend database",
    "responsive":     "responsive design mobile-first web frontend UI",
    "bootstrap":      "Bootstrap CSS framework frontend UI components",
    "scss":           "SCSS Sass CSS preprocessor styling web frontend",
}


def _augment(skill: str) -> str:
    """
    Expand a skill name with semantic context descriptors.
    Uses substring matching on lowercased skill, so one entry covers many variants.
    Falls back to the original skill if no key matches.
    """
    lower = skill.lower()
    for key, context in CONTEXT_MAP.items():
        if key in lower:
            return f"{skill} {context}"
    return skill


def _clean(s: str) -> str:
    """Normalise separators and collapse whitespace."""
    s = s.strip()
    s = re.sub(r"[_/\\-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Embedding helpers
# ─────────────────────────────────────────────────────────────────────────────

def _embed(texts: List[str]) -> np.ndarray:
    """Return L2-normalised embedding matrix (cosine sim = dot product)."""
    model = _get_model()
    vecs  = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return vecs / norms


def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between all row pairs (both already L2-normalised)."""
    return a @ b.T


# ─────────────────────────────────────────────────────────────────────────────
# Proficiency helpers
# ─────────────────────────────────────────────────────────────────────────────

PROFICIENCY_MAP: Dict[str, int] = {
    "Beginner":     1,
    "Intermediate": 2,
    "Advanced":     3,
    "Expert":       4,
}
PROFICIENCY_REVERSE: Dict[int, str] = {v: k for k, v in PROFICIENCY_MAP.items()}


def proficiency_multiplier(candidate_level: int, required_level: int) -> float:
    """
    Credit fraction based on proficiency gap.
      candidate >= required  → 1.00  (full credit)
      candidate == req - 1   → 0.75
      candidate == req - 2   → 0.45
      candidate <  req - 2   → 0.20  (minimal partial credit)
    """
    diff = candidate_level - required_level
    if diff >= 0:    return 1.00
    elif diff == -1: return 0.75
    elif diff == -2: return 0.45
    else:            return 0.20


# ─────────────────────────────────────────────────────────────────────────────
# SkillMatcher
# ─────────────────────────────────────────────────────────────────────────────

class SkillMatcher:
    """
    Semantic skill matcher backed by all-MiniLM-L6-v2.

    Two-pass matching strategy
    --------------------------
    Pass 1 (augmented): encode both query and corpus with context descriptors.
                        Best for abstract → tool matches:
                        "browser automation" → "Selenium …automation…"
    Pass 2 (raw):       encode both as raw cleaned names.
                        Best for near-exact variant matches:
                        "ReactJS" → "React.js"
    Final score = max(pass1_score, pass2_score)

    Parameters
    ----------
    known_skills : list[str]
        Candidate's skill names (any format).
    threshold : float
        Minimum cosine similarity to accept a match.
        Default 0.40 — works well with context augmentation.
    """

    DEFAULT_THRESHOLD = 0.40

    def __init__(self, known_skills: List[str], threshold: float = DEFAULT_THRESHOLD):
        self.threshold       = threshold
        self.original_skills = list(known_skills)

        if not known_skills:
            self._aug_emb = np.zeros((0, 384))
            self._raw_emb = np.zeros((0, 384))
            return

        augmented = [_augment(_clean(s)) for s in known_skills]
        raw       = [_clean(s) for s in known_skills]

        self._aug_emb = _embed(augmented)   # (N, 384)
        self._raw_emb = _embed(raw)         # (N, 384)

    # ── Single query ─────────────────────────────────────────────────────────

    def best_match(self, query: str) -> Tuple[Optional[str], float]:
        """
        Find the best matching skill from the candidate's skill set.

        Returns
        -------
        (matched_name, similarity)  or  (None, best_score) if below threshold.
        """
        if len(self.original_skills) == 0:
            return None, 0.0

        aug_q = _embed([_augment(_clean(query))])   # (1, 384)
        raw_q = _embed([_clean(query)])              # (1, 384)

        aug_sims = _cosine_sim_matrix(aug_q, self._aug_emb)[0]  # (N,)
        raw_sims = _cosine_sim_matrix(raw_q, self._raw_emb)[0]  # (N,)
        combined = np.maximum(aug_sims, raw_sims)                # (N,)

        idx   = int(np.argmax(combined))
        score = float(combined[idx])

        if score >= self.threshold:
            return self.original_skills[idx], score
        return None, score

    # ── Batch query ───────────────────────────────────────────────────────────

    def match_all(
        self,
        queries: List[str],
        threshold: Optional[float] = None,
    ) -> Dict[str, Tuple[Optional[str], float]]:
        """
        Match multiple required skills in a single batched encode call.

        Returns
        -------
        {query → (best_match_name, similarity)}
        """
        t = threshold if threshold is not None else self.threshold

        if not queries or len(self.original_skills) == 0:
            return {q: (None, 0.0) for q in queries}

        aug_q_emb = _embed([_augment(_clean(q)) for q in queries])  # (M, 384)
        raw_q_emb = _embed([_clean(q) for q in queries])            # (M, 384)

        aug_sims = _cosine_sim_matrix(aug_q_emb, self._aug_emb)     # (M, N)
        raw_sims = _cosine_sim_matrix(raw_q_emb, self._raw_emb)     # (M, N)
        combined = np.maximum(aug_sims, raw_sims)                    # (M, N)

        results: Dict[str, Tuple[Optional[str], float]] = {}
        for i, query in enumerate(queries):
            idx   = int(np.argmax(combined[i]))
            score = float(combined[i, idx])
            if score >= t:
                results[query] = (self.original_skills[idx], score)
            else:
                results[query] = (None, score)

        return results

    # ── Utility ───────────────────────────────────────────────────────────────

    def similarity(self, skill_a: str, skill_b: str) -> float:
        """Cosine similarity between two arbitrary skill strings (augmented)."""
        vecs = _embed([_augment(_clean(skill_a)), _augment(_clean(skill_b))])
        return float(_cosine_sim_matrix(vecs[[0]], vecs[[1]])[0, 0])


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    candidate_skills = [
        "Node.js", "React.js", "Python", "PostgreSQL", "Docker",
        "WebSockets", "JWT", "Express.js", "Django", "CI/CD",
        "AutoCAD", "Selenium", "OpenAI API",
    ]

    job_required = [
        "Node",               # → Node.js       ✓
        "ReactJS",            # → React.js      ✓
        "MongoDB",            # → NO MATCH      ✓  (not in candidate skills)
        "REST APIs",          # → Express.js    ✓  (augmented: REST API backend)
        "CAD software",       # → AutoCAD       ✓
        "Postgres",           # → PostgreSQL    ✓
        "browser automation", # → Selenium      ✓  (augmented: browser automation)
        "LLM integration",    # → OpenAI API    ✓  (augmented: LLM AI integration)
    ]

    print("=" * 60)
    print(f"Model     : {_MODEL_NAME}")
    print(f"Threshold : {SkillMatcher.DEFAULT_THRESHOLD}")
    print("=" * 60)

    matcher = SkillMatcher(candidate_skills)
    results = matcher.match_all(job_required)

    for query, (match, score) in results.items():
        status     = "✓" if match else "✗"
        match_str  = f"{match} ({score:.3f})" if match else f"NO MATCH ({score:.3f})"
        print(f"  {status}  '{query}' → {match_str}")
