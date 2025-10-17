#!/usr/bin/env python3
"""
app.py - Enhanced with LangChain Agents

FastAPI backend for Contextual Personal Assistant with dual LangChain agents:
1. Ingestion Agent - Enhanced entity extraction with LLM
2. Thinking Agent - AI-powered insights generation

Setup:
  1. Create a .env file with: OPENAI_API_KEY=sk-your-key-here
  2. pip install -r requirements.txt
  3. python -m spacy download en_core_web_sm
  
Run:
  uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

import os
import sqlite3
import json
import re
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, date, timedelta

import numpy as np
import dateparser
import spacy
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Modern LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# ==================== Configuration ====================
DB_PATH = "assistant.db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
ENVELOPE_SIMILARITY_THRESHOLD = 0.50

# ==================== Load Models ====================
print("Loading models...")
try:
    nlp = spacy.load("en_core_web_sm")
    print("[OK] spaCy model loaded")
except Exception as e:
    raise RuntimeError("Install spaCy model: python -m spacy download en_core_web_sm") from e

try:
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"[OK] Sentence transformer loaded: {EMBEDDING_MODEL_NAME}")
except Exception as e:
    raise RuntimeError(f"Failed to load embedding model {EMBEDDING_MODEL_NAME}") from e

# ==================== Pydantic Models ====================
class NoteRequest(BaseModel):
    note: str

class CardResponse(BaseModel):
    card_id: int
    created_new_envelope: bool
    envelope_id: int
    envelope_score: float
    card_type: str
    extracted_date: Optional[str] = None
    extracted_time: Optional[str] = None

class Envelope(BaseModel):
    id: int
    name: str
    topic_keywords: List[str]
    created_at: str
    card_count: Optional[int] = 0

class Card(BaseModel):
    id: int
    type: str
    description: str
    date: Optional[str] = None
    time: Optional[str] = None
    assignee: Optional[str] = None
    context_keywords: List[str]
    envelope_id: Optional[int] = None
    created_at: str

class UserContext(BaseModel):
    active_projects: List[str]
    contacts: List[str]
    upcoming_deadlines: List[str]
    themes: List[str]

class ThinkingResponse(BaseModel):
    insights: Dict[str, Any]
    natural_text: str

# ==================== Database Functions ====================
def init_db(path: str = DB_PATH):
    """Initialize SQLite database with required tables."""
    con = sqlite3.connect(path)
    cur = con.cursor()
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS cards (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT NOT NULL,
        description TEXT NOT NULL,
        date TEXT,
        time TEXT,
        assignee TEXT,
        context_keywords TEXT,
        envelope_id INTEGER,
        embedding BLOB,
        created_at TEXT NOT NULL
    )
    """)
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS envelopes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        topic_keywords TEXT,
        embedding BLOB,
        created_at TEXT NOT NULL
    )
    """)
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_context (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        context_json TEXT NOT NULL
    )
    """)
    
    initial_context = {
        "active_projects": [],
        "contacts": [],
        "upcoming_deadlines": [],
        "themes": []
    }
    cur.execute(
        "INSERT OR IGNORE INTO user_context (id, context_json) VALUES (1, ?)",
        (json.dumps(initial_context),)
    )
    
    con.commit()
    con.close()
    print("[OK] Database initialized")

def to_bytes(np_array: np.ndarray) -> bytes:
    return np_array.tobytes()

def from_bytes(b: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(b, dtype=dtype)

# ==================== NLP Processing Functions ====================
def preprocess_text(text: str) -> str:
    """Clean and normalize input text."""
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty")
    return " ".join(text.strip().split())

def extract_time(text: str) -> Optional[str]:
    """Extract time information from text, preserving am/pm indicator."""
    patterns = [
        r'\b(\d{1,2})\.(\d{2})\s*(a\.m\.|p\.m\.|AM|PM|am|pm)\b',
        r'\b(\d{1,2}):(\d{2})\s*(a\.m\.|p\.m\.|AM|PM|am|pm)\b',
        r'\b(\d{1,2})\s*(a\.m\.|p\.m\.|AM|PM|am|pm)\b',
        r'\b(\d{1,2}):(\d{2})(?!\s*(?:a\.m\.|p\.m\.|AM|PM|am|pm))\b',
        r'\b(\d{1,2})\.(\d{2})(?!\s*(?:a\.m\.|p\.m\.|AM|PM|am|pm))\b',
    ]
    
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            time_str = m.group(0)
            has_ampm = any(indicator in time_str.lower() for indicator in ['am', 'pm', 'a.m', 'p.m'])
            
            if has_ampm:
                if 'AM' in time_str:
                    time_str = time_str.replace('AM', 'a.m')
                elif 'am' in time_str and 'a.m' not in time_str:
                    time_str = time_str.replace('am', 'a.m')
                elif 'PM' in time_str:
                    time_str = time_str.replace('PM', 'p.m')
                elif 'pm' in time_str and 'p.m' not in time_str:
                    time_str = time_str.replace('pm', 'p.m')
                return time_str
            else:
                parts = time_str.replace(':', '.').split('.')
                if len(parts) == 2 and len(parts[1]) == 2:
                    return time_str
    
    return None

def parse_relative_date(text: str, base_date: date) -> Optional[date]:
    """Parse relative dates like 'tomorrow', 'next Monday', etc."""
    lower = text.lower()
    
    if any(phrase in lower for phrase in ['today', 'this evening', 'tonight']):
        return base_date
    
    if 'tomorrow' in lower:
        return base_date + timedelta(days=1)
    
    if 'next week' in lower:
        return base_date + timedelta(days=7)
    
    days_map = {
        'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
        'friday': 4, 'saturday': 5, 'sunday': 6
    }
    
    for day_name, day_num in days_map.items():
        if day_name in lower:
            current_weekday = base_date.weekday()
            
            if 'next' in lower:
                days_ahead = (day_num - current_weekday) % 7
                if days_ahead == 0:
                    days_ahead = 7
            else:
                days_ahead = (day_num - current_weekday) % 7
                if days_ahead == 0:
                    days_ahead = 7
            
            return base_date + timedelta(days=days_ahead)
    
    return None

def detect_card_type(text: str, has_temporal: bool) -> str:
    """Classify card type based on content."""
    doc = nlp(text)
    text_lower = text.lower()
    
    reminder_keywords = {"remember", "remind", "don't forget", "pickup", "pick up"}
    if any(k in text_lower for k in reminder_keywords):
        return "Reminder"
    
    action_verbs = {
        "call", "email", "schedule", "meet", "buy", "research",
        "create", "send", "finish", "review", "write", "prepare"
    }
    for tok in doc:
        if tok.pos_ == "VERB" and tok.lemma_.lower() in action_verbs:
            return "Task"
    
    if has_temporal:
        return "Reminder"
    
    return "Idea/Note"

def extract_entities(text: str) -> Tuple[Optional[str], Optional[str], Optional[str], List[str], bool]:
    """Extract entities from text using spaCy NER."""
    doc = nlp(text)
    base_date = date.today()
    
    assignee = None
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            if ent.start > 0:
                prev_token = doc[ent.start - 1]
                action_verbs = {'call', 'email', 'meet', 'contact', 'message', 'text'}
                if prev_token.lemma_.lower() in action_verbs:
                    assignee = ent.text
                else:
                    assignee = ent.text
            else:
                assignee = ent.text
            break
    
    if not assignee:
        for ent in doc.ents:
            if ent.label_ == "ORG":
                assignee = ent.text
                break
    
    has_temporal = any(ent.label_ in ("DATE", "TIME") for ent in doc.ents)
    
    time_str = extract_time(text)
    if time_str:
        has_temporal = True
    
    settings = {
        'PREFER_DATES_FROM': 'future',
        'RELATIVE_BASE': datetime.combine(base_date, datetime.min.time())
    }
    
    parsed = dateparser.parse(text, settings=settings)
    date_iso = None
    
    if parsed:
        date_iso = parsed.date().isoformat()
        has_temporal = True
    else:
        date_entities = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
        for date_text in date_entities:
            parsed = dateparser.parse(date_text, settings=settings)
            if parsed:
                date_iso = parsed.date().isoformat()
                has_temporal = True
                break
        
        if not date_iso:
            rel = parse_relative_date(text, base_date)
            if rel:
                date_iso = rel.isoformat()
                has_temporal = True
    
    keywords = set()
    temporal_keywords = {
        'today', 'tomorrow', 'tonight', 'week', 'month',
        'next', 'this', 'last', 'yesterday', 'friday', 'monday',
        'tuesday', 'wednesday', 'thursday', 'saturday', 'sunday'
    }
    
    for ent in doc.ents:
        if ent.label_ not in ("DATE", "TIME") and len(ent.text) > 2:
            k = ent.text.lower()
            if k not in temporal_keywords:
                keywords.add(k)
    
    for tok in doc:
        if tok.pos_ in ("NOUN", "PROPN") and not tok.is_stop and len(tok.lemma_) > 2:
            k = tok.lemma_.lower()
            if k not in temporal_keywords:
                keywords.add(k)
    
    return date_iso, time_str, assignee, sorted(list(keywords)), has_temporal

def compute_embedding(text: str) -> np.ndarray:
    """Generate semantic embedding for text."""
    emb = embed_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return emb.astype(np.float32)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))

# ==================== Database Operations ====================
def insert_card(card: Dict[str, Any]) -> int:
    """Insert a new card into the database."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
    INSERT INTO cards (type, description, date, time, assignee, 
                      context_keywords, envelope_id, embedding, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        card.get("type"),
        card.get("description"),
        card.get("date"),
        card.get("time"),
        card.get("assignee"),
        json.dumps(card.get("context_keywords", [])),
        card.get("envelope_id"),
        to_bytes(card["embedding"]),
        datetime.utcnow().isoformat()
    ))
    con.commit()
    rowid = cur.lastrowid
    con.close()
    return rowid

def get_envelopes() -> List[Dict[str, Any]]:
    """Retrieve all envelopes from database."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT id, name, topic_keywords, embedding, created_at FROM envelopes")
    rows = cur.fetchall()
    
    envs = []
    for r in rows:
        emb = from_bytes(r[3]) if r[3] else None
        envs.append({
            "id": r[0],
            "name": r[1],
            "topic_keywords": json.loads(r[2]) if r[2] else [],
            "embedding": emb,
            "created_at": r[4]
        })
    con.close()
    return envs

def insert_envelope(name: str, topic_keywords: List[str], embedding: np.ndarray) -> int:
    """Create a new envelope."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO envelopes (name, topic_keywords, embedding, created_at) VALUES (?, ?, ?, ?)",
        (name, json.dumps(topic_keywords), to_bytes(embedding), datetime.utcnow().isoformat())
    )
    con.commit()
    nid = cur.lastrowid
    con.close()
    return nid

def get_cards(envelope_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Retrieve cards, optionally filtered by envelope."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    
    if envelope_id is None:
        cur.execute("""
        SELECT id, type, description, date, time, assignee, 
               context_keywords, envelope_id, created_at 
        FROM cards ORDER BY created_at DESC
        """)
    else:
        cur.execute("""
        SELECT id, type, description, date, time, assignee, 
               context_keywords, envelope_id, created_at 
        FROM cards WHERE envelope_id = ? ORDER BY created_at DESC
        """, (envelope_id,))
    
    rows = cur.fetchall()
    cards = []
    for r in rows:
        cards.append({
            "id": r[0],
            "type": r[1],
            "description": r[2],
            "date": r[3],
            "time": r[4],
            "assignee": r[5],
            "context_keywords": json.loads(r[6]) if r[6] else [],
            "envelope_id": r[7],
            "created_at": r[8]
        })
    con.close()
    return cards

def get_user_context() -> Dict[str, Any]:
    """Retrieve current user context."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT context_json FROM user_context WHERE id=1")
    row = cur.fetchone()
    con.close()
    
    if row:
        return json.loads(row[0])
    return {
        "active_projects": [],
        "contacts": [],
        "upcoming_deadlines": [],
        "themes": []
    }

def update_user_context(updates: Dict[str, Any]):
    """Update user context with new information."""
    ctx = get_user_context()
    
    for k, v in updates.items():
        if isinstance(v, list):
            existing = ctx.get(k, [])
            seen = set(existing)
            for it in v:
                if it not in seen:
                    existing.append(it)
                    seen.add(it)
            ctx[k] = existing
        else:
            ctx[k] = v
    
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "UPDATE user_context SET context_json = ? WHERE id = 1",
        (json.dumps(ctx),)
    )
    con.commit()
    con.close()

# ==================== LANGCHAIN AGENT 1: INGESTION AGENT ====================
class IngestionAgent:
    """LangChain agent for enhanced entity extraction and classification."""
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found in environment")
        
        self.llm = ChatOpenAI(
            temperature=0.1,
            model="gpt-4o-mini",
            api_key=api_key
        )
        self.parser = StrOutputParser()
        print("[OK] Ingestion Agent initialized with GPT-4o-mini")
    
    def enhance_extraction(self, text: str, base_extraction: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to enhance or correct the base spaCy extraction."""
        
        today = date.today().isoformat()
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        
        prompt = ChatPromptTemplate.from_template("""You are an expert at extracting structured information from notes.

Given a note and preliminary extraction, refine or correct the information.

Note: "{text}"

Preliminary extraction:
- Card Type: {card_type}
- Date: {date_val}
- Time: {time_val}
- Assignee: {assignee}
- Keywords: {keywords}

Today: {today}
Tomorrow: {tomorrow}

Review and return ONLY valid JSON with corrections/enhancements:
{{
  "card_type": "Task|Reminder|Idea/Note",
  "description": "cleaned concise description",
  "date": "YYYY-MM-DD or null",
  "time": "time string or null",
  "assignee": "person/org or null",
  "context_keywords": ["keyword1", "keyword2"]
}}

Rules:
- If "Call Sarah", assignee should be "Sarah" not "Call Sarah"
- Remove temporal words from keywords
- Keep 2-5 meaningful keywords
""")
        
        chain = prompt | self.llm | self.parser
        
        try:
            response = chain.invoke({
                "text": text,
                "card_type": base_extraction.get("card_type", "Idea/Note"),
                "date_val": base_extraction.get("date") or "null",
                "time_val": base_extraction.get("time") or "null",
                "assignee": base_extraction.get("assignee") or "null",
                "keywords": ", ".join(base_extraction.get("keywords", [])),
                "today": today,
                "tomorrow": tomorrow
            })
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                enhanced = json.loads(json_match.group())
                print(f"[LLM Enhanced] {enhanced}")
                return enhanced
            else:
                return base_extraction
                
        except Exception as e:
            print(f"[WARNING] LLM enhancement failed: {e}, using base extraction")
            return base_extraction
    
    def suggest_envelope(self, card_data: Dict[str, Any], envelopes: List[Dict]) -> Tuple[Optional[int], str]:
        """Use LLM to suggest best envelope."""
        
        if not envelopes:
            return None, "No existing envelopes"
        
        meaningful_envs = [e for e in envelopes if e.get('topic_keywords') and len(e['topic_keywords']) > 0]
        
        if not meaningful_envs:
            misc_env = next((e for e in envelopes if 'miscellaneous' in e['name'].lower()), None)
            if misc_env:
                card_keywords = card_data.get("context_keywords", [])
                if not card_keywords or len(card_keywords) == 0:
                    return misc_env['id'], "Using existing Miscellaneous envelope"
            return None, "No meaningful envelopes found"
        
        env_summary = "\n".join([
            f"- ID {e['id']}: {e['name']} (keywords: {', '.join(e['topic_keywords'][:3])})"
            for e in meaningful_envs[:10]
        ])
        
        card_keywords = card_data.get("context_keywords", [])
        if not card_keywords:
            return None, "No keywords to match"
        
        prompt = ChatPromptTemplate.from_template("""You are an expert at organizing tasks into projects/envelopes.

Existing Envelopes:
{env_summary}

Card to organize:
- Description: {description}
- Keywords: {keywords}
- Assignee: {assignee}

Rules:
1. Match based on semantic similarity of keywords and description
2. If keywords overlap significantly (2+ common keywords), assign to that envelope
3. If assignee matches an envelope name/keywords, strongly prefer that envelope
4. Only create new envelope if this card is truly about a different topic
5. Confidence >= 0.6 means assign to existing, < 0.6 means create new

Return ONLY valid JSON:
{{
  "envelope_id": <number or null>,
  "confidence": <0.0 to 1.0>,
  "reasoning": "brief explanation"
}}
""")
        
        chain = prompt | self.llm | self.parser
        
        try:
            response = chain.invoke({
                "env_summary": env_summary,
                "description": card_data.get("description", ""),
                "keywords": ", ".join(card_keywords),
                "assignee": card_data.get("assignee") or "None"
            })
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                env_id = result.get("envelope_id")
                confidence = result.get("confidence", 0.5)
                reasoning = result.get("reasoning", "")
                
                print(f"[LLM Envelope] ID={env_id}, Conf={confidence}, Reason={reasoning}")
                
                if env_id and confidence >= 0.6:
                    if any(e["id"] == env_id for e in meaningful_envs):
                        return env_id, reasoning
                
                return None, reasoning
            else:
                return None, "Failed to parse LLM response"
                
        except Exception as e:
            print(f"[WARNING] LLM envelope suggestion failed: {e}")
            return None, str(e)

_ingestion_agent = None

def get_ingestion_agent():
    """Get or create ingestion agent."""
    global _ingestion_agent
    if _ingestion_agent is None:
        _ingestion_agent = IngestionAgent()
    return _ingestion_agent

# ==================== Enhanced Processing Pipeline ====================
def process_note_with_llm(text: str) -> Dict[str, Any]:
    """Enhanced pipeline with LangChain Ingestion Agent."""
    
    text_clean = preprocess_text(text)
    
    date_iso, time_str, assignee, keywords, has_temporal = extract_entities(text_clean)
    card_type = detect_card_type(text_clean, has_temporal)
    
    base_extraction = {
        "card_type": card_type,
        "description": text_clean,
        "date": date_iso,
        "time": time_str,
        "assignee": assignee,
        "keywords": keywords
    }
    
    try:
        agent = get_ingestion_agent()
        enhanced = agent.enhance_extraction(text_clean, base_extraction)
        
        card_type = enhanced.get("card_type", card_type)
        description = enhanced.get("description", text_clean)
        date_iso = enhanced.get("date", date_iso)
        time_str = enhanced.get("time", time_str)
        assignee = enhanced.get("assignee", assignee)
        keywords = enhanced.get("context_keywords", keywords)
        
        keywords = [k for k in keywords if k and len(k) > 1]
        
    except Exception as e:
        print(f"[INFO] Using base extraction (LLM unavailable): {e}")
        description = text_clean
        keywords = [k for k in keywords if k and len(k) > 1]
    
    emb_text = description + " " + " ".join(keywords)
    embedding = compute_embedding(emb_text)
    
    envelopes = get_envelopes()
    env_id = None
    score = 0.0
    created_new = False
    
    if not keywords or len(keywords) == 0:
        misc_env = next((e for e in envelopes if 'miscellaneous' in e['name'].lower()), None)
        if misc_env:
            env_id = misc_env['id']
            score = 0.5
            print(f"[Envelope] Using existing Miscellaneous (no keywords)")
        else:
            env_id = insert_envelope("Miscellaneous", ["general"], embedding)
            created_new = True
            score = 1.0
            print(f"[Envelope] Created Miscellaneous (no keywords)")
    else:
        try:
            agent = get_ingestion_agent()
            suggested_id, reasoning = agent.suggest_envelope({
                "description": description,
                "context_keywords": keywords,
                "assignee": assignee
            }, envelopes)
            
            if suggested_id and any(e["id"] == suggested_id for e in envelopes):
                env_id = suggested_id
                score = 0.8
                print(f"[LLM Envelope] Assigned to {env_id}: {reasoning}")
            else:
                print(f"[LLM Envelope] Suggests new envelope: {reasoning}")
                env_id = None
                score = 0.0
        except Exception as e:
            print(f"[INFO] LLM envelope suggestion unavailable: {e}")
            env_id = None
            score = 0.0
        
        if env_id is None:
            best_id, best_score = find_best_envelope_semantic(embedding, keywords, envelopes)
            
            if best_score >= 0.65:
                env_id = best_id
                score = best_score
                env_name = next((e['name'] for e in envelopes if e['id'] == best_id), "Unknown")
                print(f"[Semantic] Assigned to {env_id} ({env_name}) with score {best_score:.2f}")
            else:
                name = generate_envelope_name_from_keywords(keywords)
                
                existing_names = [e['name'].lower() for e in envelopes]
                if name.lower() in existing_names:
                    if assignee:
                        name = f"{name} & {assignee}"
                    else:
                        name = f"{name} Tasks"
                
                env_id = insert_envelope(name, keywords, embedding)
                created_new = True
                score = 1.0
                print(f"[New Envelope] Created '{name}' with keywords: {keywords}")
    
    card = {
        "type": card_type,
        "description": description,
        "date": date_iso,
        "time": time_str,
        "assignee": assignee,
        "context_keywords": keywords,
        "envelope_id": env_id,
        "embedding": embedding
    }
    card_id = insert_card(card)
    
    context_updates = {}
    if assignee:
        context_updates["contacts"] = [assignee]
    if date_iso:
        context_updates["upcoming_deadlines"] = [date_iso]
    if keywords:
        context_updates["themes"] = keywords[:5]
        envs = get_envelopes()
        env_name = next((e["name"] for e in envs if e["id"] == env_id), None)
        if env_name and 'miscellaneous' not in env_name.lower():
            context_updates["active_projects"] = [env_name]
    
    if context_updates:
        update_user_context(context_updates)
    
    return {
        "card_id": card_id,
        "created_new_envelope": created_new,
        "envelope_id": env_id,
        "envelope_score": score,
        "card_type": card_type,
        "extracted_date": date_iso,
        "extracted_time": time_str
    }

def find_best_envelope_semantic(embedding: np.ndarray, keywords: List[str], envelopes: List[Dict]) -> Tuple[Optional[int], float]:
    """Find best envelope using semantic similarity."""
    if not envelopes:
        return None, 0.0
    
    best_id = None
    best_score = 0.0
    
    for env in envelopes:
        env_emb = env.get("embedding")
        emb_score = cosine_similarity(embedding, env_emb) if env_emb is not None else 0.0
        
        env_keywords = set([k.lower() for k in env.get("topic_keywords", [])])
        card_keywords = set([k.lower() for k in keywords])
        keyword_score = 0.0
        
        if env_keywords and card_keywords:
            intersection = len(env_keywords.intersection(card_keywords))
            union = max(len(env_keywords), len(card_keywords))
            keyword_score = intersection / union
        
        combined = 0.7 * emb_score + 0.3 * keyword_score
        
        if combined > best_score:
            best_score = combined
            best_id = env["id"]
    
    return best_id, best_score

def generate_envelope_name_from_keywords(keywords: List[str]) -> str:
    """Generate a meaningful envelope name from keywords."""
    if not keywords:
        return "Miscellaneous"
    
    temporal = {'today', 'tomorrow', 'next', 'this', 'week', 'month'}
    filtered = [k for k in keywords if k not in temporal]
    
    if not filtered:
        filtered = keywords
    
    meaningful = sorted(filtered, key=len, reverse=True)[:2]
    
    if not meaningful:
        meaningful = keywords[:2]
    
    return " & ".join([m.title() for m in meaningful])

# ==================== LANGCHAIN AGENT 2: THINKING AGENT ====================
class ThinkingAgent:
    """LangChain agent for generating natural language insights."""
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found in environment")
        
        self.llm = ChatOpenAI(
            temperature=0.2,
            model="gpt-4o-mini",
            api_key=api_key
        )
        self.parser = StrOutputParser()
        print("[OK] Thinking Agent initialized with GPT-4o-mini")
    
    def generate_insights(self, insights_dict: dict) -> str:
        """Convert structured insights to natural language with beautiful formatting."""
        
        prompt = ChatPromptTemplate.from_template("""You are an AI assistant that creates beautiful, actionable task summaries.

Given structured insights data, create a well-formatted report with emojis and clear sections.

IMPORTANT FORMATTING RULES:
1. Start with: 🤔 Enhanced Thinking Agent Insights:
2. Use these section headers WITH emojis:
   - ⚡ Priority Tasks (Next 3 Days): for upcoming tasks
   - 🔴 Overdue Tasks: for overdue items
   - ⚠️ Scheduling Conflicts: for conflicts
   - 🔄 Potential Duplicate Tasks: for duplicates
   - 💡 Suggested Next Steps: for next steps
   - 🔗 Merge Suggestions: for envelope merges
3. For priority tasks, format as: [Urgency] Description Due: Date
4. For duplicates, show: "Task 1" vs "Task 2" Similarity: XX%
5. Be concise and actionable
6. Only include sections that have data

Insights Data:
{insights_json}

Generate a formatted summary now:""")
        
        chain = prompt | self.llm | self.parser
        
        try:
            response = chain.invoke({
                "insights_json": json.dumps(insights_dict, indent=2)
            })
            return response
        except Exception as e:
            print(f"[ERROR] LLM summary generation failed: {e}")
            import traceback
            traceback.print_exc()
            return self._generate_fallback_summary(insights_dict)
    
    def _generate_fallback_summary(self, insights: dict) -> str:
        """Generate a manual fallback summary if LLM fails."""
        lines = ["🤔 Enhanced Thinking Agent Insights:\n"]
        
        # Priority Tasks
        if insights.get("priority_tasks"):
            lines.append("⚡ Priority Tasks (Next 3 Days):")
            for task in insights["priority_tasks"][:5]:
                urgency = task.get("urgency", "Medium")
                desc = task.get("description", "")
                due = task.get("due_date", "")
                days = task.get("days_until", 0)
                
                if days == 0:
                    due_text = "Due: TODAY"
                elif days == 1:
                    due_text = f"Due: tomorrow ({due})"
                else:
                    due_text = f"Due: {due}"
                
                lines.append(f"  • [{urgency}] {desc} {due_text}")
            lines.append("")
        
        # Scheduling Conflicts
        if insights.get("conflicts"):
            lines.append("⚠️ Scheduling Conflicts:")
            for conflict in insights["conflicts"][:3]:
                assignee = conflict.get("assignee", "Unknown")
                date = conflict.get("date", "")
                tasks = conflict.get("tasks", [])
                task_list = " - ".join([t.get("description", "")[:50] for t in tasks[:2]])
                lines.append(f"  • {assignee} on {date}: {task_list}")
            lines.append("")
        
        # Potential Duplicates
        if insights.get("potential_duplicates"):
            lines.append("🔄 Potential Duplicate Tasks:")
            shown = 0
            for dup in insights["potential_duplicates"]:
                if shown >= 5:
                    break
                sim = int(dup.get("similarity", 0) * 100)
                if sim >= 80:  # Only show high similarity
                    card_a = dup.get("card_a", {}).get("description", "")[:50]
                    card_b = dup.get("card_b", {}).get("description", "")[:50]
                    lines.append(f"  • \"{card_a}\" vs \"{card_b}\" Similarity: {sim}%")
                    shown += 1
            if shown > 0:
                lines.append("")
        
        # Next Steps
        if insights.get("next_steps"):
            lines.append("💡 Suggested Next Steps:")
            for step in insights["next_steps"][:5]:
                env_name = step.get("envelope", {}).get("name", "")
                lines.append(f"  • Consider documenting learnings for '{env_name}'")
            lines.append("")
        
        # Overdue Tasks
        if insights.get("overdue_tasks"):
            lines.append("🔴 Overdue Tasks:")
            for task in insights["overdue_tasks"][:3]:
                desc = task.get("description", "")
                days = task.get("days_overdue", 0)
                lines.append(f"  • {desc} - {days} days overdue")
            lines.append("")
        
        return "\n".join(lines)

_thinking_agent = None

def get_thinking_agent():
    """Get or create thinking agent."""
    global _thinking_agent
    if _thinking_agent is None:
        _thinking_agent = ThinkingAgent()
    return _thinking_agent

# ==================== Thinking Agent Logic ====================
def thinking_agent_run() -> Dict[str, Any]:
    """Run the thinking agent to generate insights from all cards."""
    cards = get_cards()
    envelopes = get_envelopes()
    today = date.today()
    
    insights = {
        "conflicts": [],
        "merge_suggestions": [],
        "next_steps": [],
        "overdue_tasks": [],
        "orphaned_contacts": [],
        "priority_tasks": [],
        "potential_duplicates": []
    }
    
    # 1. Detect scheduling conflicts
    tasks = [c for c in cards if c["type"] in ("Task", "Reminder") 
             and c["date"] and c.get("assignee")]
    
    date_assignee_map = {}
    for t in tasks:
        key = (t["date"], t["assignee"])
        date_assignee_map.setdefault(key, []).append(t)
    
    for k, items in date_assignee_map.items():
        if len(items) > 1:
            insights["conflicts"].append({
                "date": k[0],
                "assignee": k[1],
                "tasks": [{"id": it["id"], "description": it["description"]} 
                         for it in items]
            })
    
    # 2. Suggest envelope merges
    for i, e1 in enumerate(envelopes):
        for j in range(i + 1, len(envelopes)):
            e2 = envelopes[j]
            set1 = set([str(x).lower() for x in e1.get("topic_keywords", [])])
            set2 = set([str(x).lower() for x in e2.get("topic_keywords", [])])
            
            if not set1 or not set2:
                continue
            
            inter = set1.intersection(set2)
            overlap = len(inter) / min(len(set1), len(set2))
            
            if overlap >= 0.5:
                insights["merge_suggestions"].append({
                    "envelope_a": {"id": e1["id"], "name": e1["name"]},
                    "envelope_b": {"id": e2["id"], "name": e2["name"]},
                    "overlap": round(overlap, 2),
                    "common_keywords": list(inter)[:5]
                })
    
    # 3. Next steps suggestions
    for env in envelopes:
        env_cards = get_cards(env["id"])
        t_env = [c for c in env_cards if c["type"] in ("Task", "Reminder")]
        ideas = [c for c in env_cards if c["type"] == "Idea/Note"]
        
        if len(t_env) > 0 and len(ideas) == 0:
            insights["next_steps"].append({
                "envelope": {"id": env["id"], "name": env["name"]},
                "suggestion": f"Consider documenting learnings or planning next phase for '{env['name']}'"
            })
    
    # 4. Overdue tasks
    for card in cards:
        if card["type"] in ("Task", "Reminder") and card.get("date"):
            try:
                cd = date.fromisoformat(card["date"])
                if cd < today:
                    insights["overdue_tasks"].append({
                        "id": card["id"],
                        "description": card["description"],
                        "due_date": card["date"],
                        "days_overdue": (today - cd).days,
                        "assignee": card.get("assignee")
                    })
            except:
                pass
    
    # 5. Orphaned contacts
    all_assignees = set()
    recent = set()
    
    for card in cards:
        if card.get("assignee"):
            all_assignees.add(card["assignee"])
            try:
                created = datetime.fromisoformat(card["created_at"])
                if (datetime.utcnow() - created).days <= 7:
                    recent.add(card["assignee"])
            except:
                pass
    
    orphaned = all_assignees - recent
    for a in orphaned:
        acards = [c for c in cards if c.get("assignee") == a]
        insights["orphaned_contacts"].append({
            "assignee": a,
            "last_mention": max([c["created_at"] for c in acards]),
            "total_cards": len(acards)
        })
    
    # 6. Priority tasks (upcoming)
    for card in cards:
        if card["type"] in ("Task", "Reminder") and card.get("date"):
            try:
                cd = date.fromisoformat(card["date"])
                days_until = (cd - today).days
                
                if 0 <= days_until <= 3:
                    insights["priority_tasks"].append({
                        "id": card["id"],
                        "description": card["description"],
                        "due_date": card["date"],
                        "days_until": days_until,
                        "assignee": card.get("assignee"),
                        "urgency": "High" if days_until == 0 else "Medium"
                    })
            except:
                pass
    
    # 7. Potential duplicates
    from difflib import SequenceMatcher
    
    for i, c1 in enumerate(cards):
        for j in range(i + 1, len(cards)):
            c2 = cards[j]
            sim = SequenceMatcher(
                None,
                c1["description"].lower(),
                c2["description"].lower()
            ).ratio()
            
            if sim >= 0.7:
                insights["potential_duplicates"].append({
                    "card_a": {"id": c1["id"], "description": c1["description"]},
                    "card_b": {"id": c2["id"], "description": c2["description"]},
                    "similarity": round(sim, 2)
                })
    
    # Sort insights
    insights["overdue_tasks"].sort(key=lambda x: x["days_overdue"], reverse=True)
    insights["priority_tasks"].sort(key=lambda x: x["days_until"])
    
    return insights

# ==================== FastAPI Application ====================
app = FastAPI(
    title="Contextual Personal Assistant API",
    description="Backend API with dual LangChain agents for intelligent note processing",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize database and agents on startup."""
    init_db()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("[OK] OpenAI API key loaded from .env")
        try:
            get_ingestion_agent()
            get_thinking_agent()
            print("[OK] LangChain agents initialized successfully")
        except Exception as e:
            print(f"[WARNING] Agent initialization failed: {e}")
            print("   Falling back to base extraction")
    else:
        print("[WARNING] OPENAI_API_KEY not found in .env file")
        print("   Natural language generation will not work without it")
    
    print("[OK] API server ready")

@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "message": "Contextual Personal Assistant API v2.0 with LangChain Agents",
        "version": "2.0.0",
        "agents": {
            "ingestion": "LangChain agent for enhanced entity extraction (GPT-4o-mini)",
            "thinking": "LangChain agent for natural language insights (GPT-4o-mini)"
        },
        "endpoints": {
            "POST /add": "Add a new note (uses LangChain Ingestion Agent)",
            "GET /cards": "List all cards",
            "GET /envelopes": "List all envelopes",
            "GET /context": "Get user context",
            "GET /think": "Run thinking agent (uses LangChain)",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    api_key_status = "configured" if os.getenv("OPENAI_API_KEY") else "missing"
    
    agents_status = {
        "ingestion": "not_initialized",
        "thinking": "not_initialized"
    }
    
    global _ingestion_agent, _thinking_agent
    if _ingestion_agent:
        agents_status["ingestion"] = "ready"
    if _thinking_agent:
        agents_status["thinking"] = "ready"
    
    return {
        "status": "healthy" if api_key_status == "configured" else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "openai_api_key": api_key_status,
        "agents": agents_status
    }

@app.post("/add", response_model=CardResponse)
async def api_add(request: NoteRequest):
    """Process a new note using LangChain Ingestion Agent."""
    if not request.note.strip():
        raise HTTPException(status_code=400, detail="Note cannot be empty")
    
    try:
        result = process_note_with_llm(request.note)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing note: {str(e)}")

@app.get("/cards", response_model=List[Card])
async def api_list_cards(envelope_id: Optional[int] = Query(None, description="Filter by envelope ID")):
    """Get all cards, optionally filtered by envelope."""
    try:
        cards = get_cards(envelope_id)
        return cards
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching cards: {str(e)}")

@app.get("/envelopes", response_model=List[Envelope])
async def api_list_envelopes():
    """Get all envelopes with card counts."""
    try:
        envs = get_envelopes()
        for env in envs:
            env["card_count"] = len(get_cards(env["id"]))
        return envs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching envelopes: {str(e)}")

@app.get("/context", response_model=UserContext)
async def api_context():
    """Get current user context (projects, contacts, deadlines, themes)."""
    try:
        context = get_user_context()
        return context
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching context: {str(e)}")

@app.get("/think/debug")
async def api_think_debug():
    """Debug endpoint to see raw insights without LLM processing."""
    try:
        insights = thinking_agent_run()
        
        non_empty = {
            k: len(v) for k, v in insights.items() 
            if isinstance(v, list) and len(v) > 0
        }
        
        return {
            "insights": insights,
            "summary": {
                "total_categories": len(insights),
                "non_empty_categories": non_empty,
                "has_data": len(non_empty) > 0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/think", response_model=ThinkingResponse)
async def api_think(natural: bool = Query(True, description="Return natural language summary")):
    """Run the thinking agent to analyze all cards and generate insights."""
    try:
        # Get structured insights
        insights = thinking_agent_run()
        
        # Convert numpy arrays and other non-serializable objects to plain Python types
        insights_clean = json.loads(json.dumps(insights, default=str))
        
        # Generate natural language summary if requested
        if natural:
            try:
                agent = get_thinking_agent()
                natural_text = agent.generate_insights(insights_clean)
                
                return {
                    "insights": insights_clean,
                    "natural_text": natural_text
                }
            except Exception as e:
                print(f"[ERROR] Natural language generation failed: {e}")
                import traceback
                traceback.print_exc()
                
                # Use fallback
                agent = get_thinking_agent()
                fallback_text = agent._generate_fallback_summary(insights_clean)
                
                return {
                    "insights": insights_clean,
                    "natural_text": fallback_text
                }
        else:
            return {
                "insights": insights_clean,
                "natural_text": "Natural language generation disabled. Set natural=true to enable."
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running thinking agent: {str(e)}")

@app.delete("/cards/{card_id}")
async def api_delete_card(card_id: int):
    """Delete a specific card."""
    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute("DELETE FROM cards WHERE id = ?", (card_id,))
        con.commit()
        affected = cur.rowcount
        con.close()
        
        if affected == 0:
            raise HTTPException(status_code=404, detail="Card not found")
        
        return {"message": f"Card {card_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting card: {str(e)}")

@app.delete("/envelopes/{envelope_id}")
async def api_delete_envelope(envelope_id: int):
    """Delete an envelope and reassign its cards."""
    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        
        cur.execute("SELECT id FROM envelopes WHERE id = ?", (envelope_id,))
        if not cur.fetchone():
            con.close()
            raise HTTPException(status_code=404, detail="Envelope not found")
        
        cur.execute("SELECT id FROM envelopes WHERE name = 'Miscellaneous'")
        misc = cur.fetchone()
        
        if misc:
            misc_id = misc[0]
        else:
            misc_emb = compute_embedding("miscellaneous general")
            cur.execute(
                "INSERT INTO envelopes (name, topic_keywords, embedding, created_at) VALUES (?, ?, ?, ?)",
                ("Miscellaneous", json.dumps(["general"]), to_bytes(misc_emb), datetime.utcnow().isoformat())
            )
            misc_id = cur.lastrowid
        
        cur.execute("UPDATE cards SET envelope_id = ? WHERE envelope_id = ?", (misc_id, envelope_id))
        cur.execute("DELETE FROM envelopes WHERE id = ?", (envelope_id,))
        
        con.commit()
        con.close()
        
        return {"message": f"Envelope {envelope_id} deleted, cards moved to Miscellaneous"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting envelope: {str(e)}")

@app.post("/envelopes/cleanup")
async def api_cleanup_envelopes():
    """Cleanup duplicate Miscellaneous envelopes and merge them into one."""
    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        
        cur.execute("SELECT id, name FROM envelopes WHERE name = 'Miscellaneous'")
        misc_envs = cur.fetchall()
        
        if len(misc_envs) <= 1:
            con.close()
            return {"message": "No duplicate Miscellaneous envelopes found", "merged": 0}
        
        keep_id = misc_envs[0][0]
        merge_ids = [env[0] for env in misc_envs[1:]]
        
        for mid in merge_ids:
            cur.execute("UPDATE cards SET envelope_id = ? WHERE envelope_id = ?", (keep_id, mid))
            cur.execute("DELETE FROM envelopes WHERE id = ?", (mid,))
        
        con.commit()
        con.close()
        
        return {
            "message": f"Merged {len(merge_ids)} duplicate Miscellaneous envelope(s) into envelope {keep_id}",
            "merged": len(merge_ids),
            "kept_envelope_id": keep_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cleaning up envelopes: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)