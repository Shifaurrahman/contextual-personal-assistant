pip install -r requirements.txt

pip install spacy==3.7.3 click==8.1.7 typer==0.9.0

python -m spacy download en_core_web_sm

pip install -U sentence-transformers huggingface-hub

uvicorn app:app --reload --host 0.0.0.0 --port 8000

pip install --upgrade langchain


# Contextual Personal Assistant - LangChain Implementation

A sophisticated AI-powered personal assistant that transforms unstructured notes into a highly organized, actionable knowledge base using **LangChain agent framework**.

## 🏗️ Architecture Overview

### Design Philosophy

This implementation uses **LangChain** as the primary agent development framework for both core agents, replacing traditional rule-based NLP with LLM-powered intelligence.

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INPUT (Raw Note)                     │
│         "Call Sarah about Q3 budget next Monday"             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│         INGESTION & ORGANIZATION AGENT (LangChain)          │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  1. LLM Entity Extraction                           │   │
│  │     • Card Type Classification (Task/Reminder/Note) │   │
│  │     • Date/Time Parsing (with context awareness)    │   │
│  │     • Assignee Detection (NER via LLM)              │   │
│  │     • Keyword Extraction (semantic understanding)   │   │
│  └─────────────────────────────────────────────────────┘   │
│                     │                                         │
│                     ▼                                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  2. Envelope Assignment (LLM + Embeddings)          │   │
│  │     • Semantic matching with existing envelopes     │   │
│  │     • Context-aware grouping decisions              │   │
│  │     • Auto-create new envelopes when needed         │   │
│  └─────────────────────────────────────────────────────┘   │
│                     │                                         │
│                     ▼                                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  3. Context Management                              │   │
│  │     • Update active projects, contacts, themes      │   │
│  │     • Maintain dynamic user context for future use  │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    STRUCTURED CARD                           │
│  Type: Task                                                  │
│  Description: Call Sarah about Q3 budget                     │
│  Date: 2025-10-20                                            │
│  Assignee: Sarah                                             │
│  Keywords: [budget, q3, sarah]                               │
│  Envelope: "Budget & Sarah"                                  │
└────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              THINKING AGENT (LangChain)                      │
│         (Runs periodically or on-demand)                     │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  LLM-Powered Analysis                               │   │
│  │  • Detect scheduling conflicts                      │   │
│  │  • Suggest envelope merges                          │   │
│  │  • Identify overdue tasks                           │   │
│  │  • Recommend next steps                             │   │
│  │  • Find potential duplicates                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                     │                                         │
│                     ▼                                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Natural Language Summary Generation                │   │
│  │  • Convert structured insights to readable report   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Technology Stack

### 1. Agent Development Framework: **LangChain**

**Choice Justification:**

- **Why LangChain over LlamaIndex?**
  - **Better for agent workflows**: LangChain excels at building multi-step agent pipelines with tool calling
  - **Flexibility**: Easy to switch between different LLM providers (OpenAI, Anthropic, local models)
  - **Rich ecosystem**: Extensive tools for prompt engineering, output parsing, and chain composition
  - **Active development**: Large community and frequent updates
  
- **Why LangChain over LiteLLM?**
  - LiteLLM is primarily a **proxy/wrapper** for multiple LLM APIs
  - LangChain provides **full agent framework** with memory, tools, and chains
  - Better suited for complex multi-step reasoning tasks

**LangChain Components Used:**

1. **ChatOpenAI**: LLM interface with structured output
2. **ChatPromptTemplate**: Dynamic prompt construction with context injection
3. **JsonOutputParser**: Ensures structured JSON responses from LLM
4. **StrOutputParser**: For natural language generation
5. **Agent chains**: Sequential processing pipelines

### 2. Model Selection

#### Primary LLM: **GPT-4o-mini**

**Reasoning:**
- **Cost-effective**: ~15x cheaper than GPT-4 (~$0.15/$0.60 per 1M tokens vs $5/$15)
- **Fast**: Low latency for real-time note processing
- **Capable**: Sufficient for entity extraction, classification, and analysis tasks
- **Good JSON adherence**: Reliable structured output generation

**Alternative Options Considered:**

| Model | Pros | Cons | Use Case |
|-------|------|------|----------|
| **GPT-4** | Highest accuracy | Expensive, slower | High-stakes enterprise use |
| **Claude 3 Haiku** | Fast, good reasoning | Requires Anthropic API | Privacy-conscious deployments |
| **Llama 3.1 8B** | Free, local | Requires GPU, lower quality | Offline/air-gapped systems |
| **Mistral 7B** | Open source, fast | Needs fine-tuning | Budget-constrained projects |

#### Embedding Model: **all-MiniLM-L6-v2**

**Reasoning:**
- **Local execution**: No API costs, fast inference
- **Small size**: 80MB model, runs on CPU
- **Good performance**: 384-dimensional embeddings, adequate for semantic search
- **Trade-off**: Accuracy vs speed (acceptable for this use case)

**Scalability Considerations:**

```python
# Current: Simple in-memory comparison
# Scales to: ~1000 envelopes before performance degrades

# Future optimization for 10K+ envelopes:
# 1. Use vector database (Pinecone, Weaviate, Chroma)
# 2. Implement HNSW indexing
# 3. Batch processing with caching
```

### 3. Storage: **SQLite**

**Reasoning:**
- ✅ **Zero configuration**: No separate database server
- ✅ **ACID compliance**: Reliable transactions
- ✅ **Portable**: Single file database
- ✅ **Sufficient for MVP**: Handles 10K+ cards easily

**Schema Design:**

```sql
cards
├── id (PK)
├── type (Task/Reminder/Idea)
├── description
├── date, time
├── assignee
├── context_keywords (JSON array)
├── envelope_id (FK)
├── embedding (BLOB)
└── created_at

envelopes
├── id (PK)
├── name
├── topic_keywords (JSON array)
├── embedding (BLOB)
└── created_at

user_context
├── id (PK, always 1)
└── context_json (JSON)
    ├── active_projects
    ├── contacts
    ├── upcoming_deadlines
    └── themes
```

## 📦 Installation & Setup

### Prerequisites
- Python 3.9+
- OpenAI API key

### Step 1: Clone & Install

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model (lightweight, used for fallback only)
python -m spacy download en_core_web_sm
```

### Step 2: Configure Environment

Create a `.env` file:

```bash
OPENAI_API_KEY=sk-your-openai-api-key-here
```

### Step 3: Run the Server

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Server will start at: `http://localhost:8000`

## 🚀 API Usage

### 1. Add a Note (Ingestion Agent)

```bash
curl -X POST http://localhost:8000/add \
  -H "Content-Type: application/json" \
  -d '{"note": "Call Sarah about Q3 budget next Monday"}'
```

**Response:**
```json
{
  "card_id": 1,
  "created_new_envelope": true,
  "envelope_id": 1,
  "envelope_score": 0.95,
  "card_type": "Task",
  "extracted_date": "2025-10-20",
  "extracted_time": null
}
```

### 2. View All Cards

```bash
curl http://localhost:8000/cards
```

### 3. View Envelopes

```bash
curl http://localhost:8000/envelopes
```

### 4. Run Thinking Agent

```bash
curl http://localhost:8000/think
```

**Response:**
```json
{
  "insights": {
    "conflicts": [...],
    "priority_tasks": [...],
    "overdue_tasks": [...]
  },
  "natural_text": "🤔 Thinking Agent Insights:\n\n⚡ Priority Tasks..."
}
```

### 5. Get User Context

```bash
curl http://localhost:8000/context
```

## 🧠 Agent Deep Dive

### Ingestion Agent Implementation

**Key Innovation: Context-Aware Prompting**

```python
# The agent uses dynamic prompts that inject:
# 1. User's current context (projects, contacts, themes)
# 2. Existing envelopes for better assignment
# 3. Today's date for relative date parsing

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert assistant...
    
    User's Current Context: {context}
    Today's date: {today}
    
    Existing Envelopes: {envelopes}
    
    Parse this note and extract structured data..."""),
    ("human", "{note}")
])
```

**Benefits:**
- **Better date parsing**: "next Monday" correctly resolves based on today
- **Smarter envelope assignment**: Uses project context to group related tasks
- **Improved entity extraction**: Recognizes people from user's contact list

### Thinking Agent Implementation

**Multi-Stage Analysis Pipeline:**

1. **Data Aggregation**: Collect all cards, envelopes, and context
2. **LLM Analysis**: Single comprehensive prompt analyzing all data
3. **Structured Output**: JSON with categorized insights
4. **Natural Summary**: Second LLM call to generate readable report

**Fallback Strategy:**

```python
try:
    insights = llm.analyze(data)  # LLM-powered
except Exception:
    insights = rule_based_analysis(data)  # Fallback
```

## 💰 Cost Analysis

### Per-Note Cost (GPT-4o-mini)

**Ingestion Agent:**
- Entity extraction: ~500 tokens → **$0.0001**
- Envelope assignment: ~800 tokens → **$0.0002**
- **Total per note: $0.0003** (0.03¢)

**Thinking Agent:**
- Analysis (50 cards): ~3000 tokens → **$0.0009**
- Summary generation: ~1000 tokens → **$0.0003**
- **Total per run: $0.0012** (0.12¢)

**Monthly estimate (1000 notes/month, daily thinking):**
- Ingestion: $0.30
- Thinking: $0.36
- **Total: $0.66/month**

### Scalability Thresholds

| Scale | Notes/Month | Monthly Cost | Optimization Needed |
|-------|-------------|--------------|---------------------|
| Personal | 100-500 | $0.10-0.50 | None |
| Power User | 1K-5K | $0.50-2.50 | None |
| Team (10) | 10K-50K | $5-25 | Batch processing |
| Enterprise | 100K+ | $150+ | Local models + caching |

## 🔄 Future Enhancements

### Short-term (Next Sprint)
1. **Batch processing** for multiple notes
2. **Confidence scores** for user review
3. **Edit/update** cards via API
4. **Search** with semantic similarity

### Medium-term
1. **Vector database** integration (Chroma/Pinecone)
2. **Local LLM** support (Llama 3, Mistral)
3. **Streaming responses** for real-time UI
4. **Multi-user** support with authentication

### Long-term
1. **Proactive suggestions** (push notifications)
2. **Integration** with calendar, email
3. **Custom agent** fine-tuning
4. **Voice input** support

## 🐛 Troubleshooting

### Issue: "OPENAI_API_KEY not found"
**Solution**: Create `.env` file with your API key

### Issue: Agent fails to extract entities
**Solution**: Check LLM response in logs. Fallback to rule-based extraction automatically activates.

### Issue: Slow performance with many envelopes
**Solution**: 
```python
# In app.py, adjust threshold:
ENVELOPE_SIMILARITY_THRESHOLD = 0.60  # Higher = fewer matches
```

## 📊 Testing

```bash
# Test note processing
curl -X POST http://localhost:8000/add \
  -d '{"note": "Meeting with John tomorrow at 3pm"}'

# Verify card creation
curl http://localhost:8000/cards | jq

# Test thinking agent
curl http://localhost:8000/think | jq '.natural_text'
```

## 📝 Design Decisions Summary

| Component | Choice | Reason |
|-----------|--------|--------|
| **Agent Framework** | LangChain | Best for multi-step agent workflows |
| **LLM** | GPT-4o-mini | Cost-effective, fast, reliable |
| **Embeddings** | all-MiniLM-L6-v2 | Local, fast, sufficient quality |
| **Database** | SQLite | Simple, portable, ACID compliance |
| **API Framework** | FastAPI | Modern, async, type-safe |

## 📄 License

MIT License - Feel free to use and modify.

---

**Built with ❤️ using LangChain**