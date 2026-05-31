"""
agents.py — NAB Multi-Agent Banking Chatbot (AI Brain)
=======================================================
This file contains all the AI agent logic. Think of it as the "brain" behind
the chatbot.
 
─────────────────────────────────────────────────────
HOW A MESSAGE IS PROCESSED — FULL STEP-BY-STEP FLOW
─────────────────────────────────────────────────────
 
  Step 1 — GREETING CHECK (Python, no AI involved)
  ─────────────────────────────────────────────────
  If the user just says "hi", "hello", "g'day" etc., the code replies with a
  fixed welcome message immediately. Gemma is never called. This keeps simple
  greetings instant and avoids burning LLM time on a one-word input.
 
  Step 2 — ORCHESTRATOR SENDS THE MESSAGE TO GEMMA
  ─────────────────────────────────────────────────
  For everything else, the OrchestratorAgent passes the user's message to
  Gemma (the local AI model running via Ollama). The orchestrator's system
  prompt tells Gemma exactly one thing: figure out WHICH specialist agent
  should handle this question, then output a small JSON object like:
 
      {"route": "loans_agent", "refined_query": "What home loans does NAB offer?"}
 
  Gemma is NOT supposed to answer the banking question itself here — just
  decide where to send it. Think of Gemma acting as a smart receptionist at
  this stage: reading the question and directing it to the right desk.
 
  Step 3 — GEMMA'S RESPONSE FALLS INTO ONE OF THREE OUTCOMES
  ────────────────────────────────────────────────────────────
 
  Outcome A — Gemma outputs valid JSON   
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  The code parses the JSON, reads the "route" field, and hands the question
  off to the matching specialist agent (e.g. LoansAgent, PaymentsAgent).
  Jump to Step 4.
 
  Outcome B — Gemma asks a clarifying question   
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Sometimes the user's message is too vague to route confidently, e.g. just
  "I need help". In that case Gemma (following its instructions) replies with
  ONE clarifying question like:
 
      "No worries! Is it related to your account, a payment, or something else?"
 
  The code detects this by checking:
    - The reply is short (under 300 characters), AND
    - It ends with "?" or contains phrases like "are you asking", "do you mean",
      "can you clarify", etc.
 
  When this is detected, the clarifying question is sent straight back to the
  user as the chatbot's reply. No specialist is called. The user answers, and
  the whole flow starts again from Step 2 with more context.
 
  Outcome C — Gemma answers the question directly  
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Occasionally Gemma ignores its routing instructions and just answers the
  banking question itself, e.g. "NAB's variable rate is currently 6.2%...".
  This is bad because:
    - Gemma doesn't have access to the knowledge base at this stage.
    - It may hallucinate incorrect rates, fees, or product details.
    - The answer bypasses the specialist agent entirely.
 
  The code catches this by checking: if the response has no JSON AND doesn't
  look like a clarifying question, it must be a direct answer. We log a
  warning and immediately trigger the Force-Route fallback (Step 3b).
 
  Step 3b — FORCE-ROUTE FALLBACK (keyword scoring)
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  When Gemma fails to output JSON (Outcome C above), Python takes over and
  routes the message itself using simple keyword matching.
 
  Each agent has a list of keywords associated with its domain. For example:
    - loans_agent    → ["home loan", "mortgage", "fixed rate", "lvr", ...]
    - payments_agent → ["transfer", "bpay", "swift", "payid", "send money", ...]
 
  The user's message is checked against all keyword lists. Each match scores
  one point. The agent with the highest score wins and the question is routed
  there. If nothing matches, it defaults to the support_agent as a safe
  catch-all. This guarantees the question always reaches SOME specialist rather
  than returning an unvalidated Gemma answer.
 
  Step 4 — SPECIALIST AGENT RETRIEVES RELEVANT KNOWLEDGE
  ────────────────────────────────────────────────────────
  The chosen specialist (e.g. LoansAgent) looks up the most relevant content
  from its knowledge base — a JSON file full of real NAB banking information.
 
  Two retrieval strategies are used depending on knowledge base size:
    - Large KB (> 8,000 chars) → FAISS semantic search: finds the top 6 most
      relevant text chunks for the user's specific question. Fast and focused.
    - Small KB (≤ 8,000 chars) → Full injection: the entire KB is included in
      the prompt. Simple and sufficient for small files.
 
  Step 5 — SPECIALIST CALLS GEMMA TO GENERATE THE ANSWER
  ────────────────────────────────────────────────────────
  The specialist agent sends Gemma:
    - Its own system prompt (personality, expertise, rules)
    - The retrieved KB content (the facts to answer from)
    - The last 2 conversation turns (for follow-up question context)
    - The user's question
 
  Gemma reads all of this and generates a natural-language answer grounded
  in the KB content. It's instructed not to fabricate data — only to use
  what's in the knowledge base.
 
  Step 6 — ANSWER RETURNED TO THE BROWSER
  ────────────────────────────────────────
  The specialist's answer is saved to session history (for future context),
  then returned to app.py, which sends it as a JSON response to the browser.
 
─────────────────────────────────────────────────────
AGENT ROSTER
─────────────────────────────────────────────────────
  OrchestratorAgent   — reads intent, routes to the right specialist
  AccountsAgent       — opening/closing accounts, joint accounts, Portal Pay
  BusinessAgent       — business banking, NAB Bookkeeper, EFTPOS
  CardsInsuranceAgent — credit/debit cards, rewards, fraud, insurance
  LoansAgent          — home loans, business loans, car loans, hardship
  PaymentsAgent       — transfers, BPAY, PayID, SWIFT, FX deals
  SupportAgent        — general help, passwords, branch info, escalations
 
─────────────────────────────────────────────────────
KNOWLEDGE BASE RETRIEVAL STRATEGY
─────────────────────────────────────────────────────
  - Large KBs (> 8,000 chars) get a FAISS vector index for semantic search.
  - Small KBs are injected whole into the prompt — simpler and fast enough.
  - FAISS indexes are built once at startup and saved to disk, so subsequent
    server restarts load them instantly with zero rebuild cost.
"""

import json, os, time, logging, re
from pathlib import Path
from typing import Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.parent
DATA_DIR        = BASE_DIR / "data"
# CACHE_FILE removed — response cache disabled
SESSIONS_FILE   = BASE_DIR / "cache" / "sessions.json"
FAISS_INDEX_DIR = BASE_DIR / "cache" / "faiss_indexes"

SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ── Wipe all session history on every server restart ─────────────────────────
# Ensures no stale conversations carry over between runs
if SESSIONS_FILE.exists():
    SESSIONS_FILE.write_text("{}", encoding="utf-8")
    print("[Startup] sessions.json cleared — fresh start.")

# ─── Config ────────────────────────────────────────────────────────────────────
# Ollama runs locally and hosts the language model. These can be overridden
# via environment variables if you want to point at a different host or model.
OLLAMA_BASE_URL    = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL       = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Knowledge base files larger than this (in characters) get a FAISS index
# for semantic search. Smaller files are injected whole into the prompt.
FAISS_THRESHOLD_CHARS = 8_000   
FAISS_TOP_K           = 6     # how many chunks to retrieve per query
CHUNK_SIZE            = 600     # chars per chunk
CHUNK_OVERLAP         = 80      # overlap between chunks

# ─── JSON helpers ──────────────────────────────────────────────────────────────
def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_json(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ─── Session History ───────────────────────────────────────────────────────────
# "Session history" is the running record of everything a user has said and
# the bot has replied during the current conversation. We store it in
# sessions.json so the AI can reference recent messages for follow-up questions
def session_get(session_id: str) -> list:
    """
    Retrieve all stored messages for a given session.
    Returns an empty list if this session hasn't sent any messages yet.
    """
    return _load_json(SESSIONS_FILE).get(session_id, [])

def session_append(session_id: str, role: str, content: str):
    """
    Add a new message turn to the session history and save it to disk.
 
    role    — "user" for the customer's message, "assistant" for the AI's reply
    content — the actual message text
 
    We keep only the last 30 messages per session to avoid the file growing
    indefinitely and to stay within the model's context window.
    """
    store = _load_json(SESSIONS_FILE)
    store.setdefault(session_id, [])
    store[session_id].append({"role": role, "content": content, "ts": time.time()})
    store[session_id] = store[session_id][-30:]
    _save_json(SESSIONS_FILE, store)

def session_clear(session_id: str):
    """
    Remove all history for a session (e.g. when the user starts a new chat).
    """
    store = _load_json(SESSIONS_FILE)
    store.pop(session_id, None)
    _save_json(SESSIONS_FILE, store)

# ─── Knowledge Base Loader ─────────────────────────────────────────────────────
FILE_MAP = {
    "accounts":        "accounts.json",
    "business":        "business.json",
    "cards_insurance": "cards-insurance.json",
    "loans":           "loans.json",
    "payments":        "payments.json",
    "support":         "support.json",
}

def load_raw_kb(category: str) -> str:
    """
    Load a knowledge base file and convert it to a single plain-text string.
 
    Each record in the JSON file has a 'title' and 'content'.
    We concatenate all of them, separated by horizontal rules, into one
    big block of text that can be searched or injected into a prompt.
 
    Returns an empty string if the file is missing or unreadable.
    """
    path = DATA_DIR / FILE_MAP.get(category, "")
    if not path.exists():
        return ""
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
        parts   = []
        for r in records:
            title   = r.get("title", "")
            content = r.get("content", r.get("summary", ""))
            if title or content:
                parts.append(f"### {title}\n{content}")
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        log.error(f"KB load error for '{category}': {e}")
        return ""

# ─── FAISS Index Management ────────────────────────────────────────────────────
#
# FAISS (Facebook AI Similarity Search) lets us find the most semantically
# relevant chunks of text for a given question. For example, if a user asks
# "How much can I borrow?", FAISS will return the chunks about LVR and
# borrowing limits rather than chunks about EFTPOS terminals.
#
# The nomic-embed-text model is used to convert text into embedding vectors
# (arrays of numbers that represent meaning), and FAISS searches those vectors.
 
# A single embeddings instance is reused rather than creating a new one per call
_embeddings_instance = None

def _get_embeddings() -> OllamaEmbeddings:
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = OllamaEmbeddings(
            model=OLLAMA_EMBED_MODEL,
            base_url=OLLAMA_BASE_URL
        )
    return _embeddings_instance

def build_or_load_index(category: str, raw_text: str) -> FAISS:
    """
    Return the FAISS vector index for a knowledge base category.
 
    If a saved index already exists on disk (from a previous server run),
    load it — this is nearly instant. If not, build it fresh by:
      1. Splitting the raw text into overlapping chunks
      2. Embedding each chunk with nomic-embed-text
      3. Storing the embeddings in a FAISS index
      4. Saving the index to disk so the next startup is instant
 
    This means rebuilding only happens once per knowledge base, not every run.
    """
    index_path = FAISS_INDEX_DIR / category

    # Load from disk if already exists (zero rebuild cost)
    if index_path.exists():
        log.info(f"[FAISS] Loading '{category}' index from disk...")
        t0    = time.time()
        index = FAISS.load_local(
            str(index_path),
            _get_embeddings(),
            allow_dangerous_deserialization=True,
        )
        log.info(f"[FAISS] '{category}' loaded in {int((time.time()-t0)*1000)}ms")
        return index

    # Build fresh index
    log.info(f"[FAISS] Building '{category}' index ({len(raw_text):,} chars)...")
    t0 = time.time()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n---\n\n", "\n\n", "\n", ". ", " "],
    )
    docs  = splitter.create_documents([raw_text], metadatas=[{"category": category}])
    index = FAISS.from_documents(docs, _get_embeddings())
    index.save_local(str(index_path))

    log.info(f"[FAISS] '{category}': {len(docs)} chunks | built & saved in {int((time.time()-t0)*1000)}ms")
    return index

def faiss_search(index: FAISS, query: str, k: int = FAISS_TOP_K) -> tuple:
    """
    Search the FAISS index and return the most relevant text chunks.
 
    Returns a tuple of:
        (context_str, retrieval_ms)
        - context_str  : the top-K chunks joined into a single string
        - retrieval_ms : how long the search took in milliseconds
    """
    t0   = time.time()
    docs = index.similarity_search(query, k=k)
    ms   = int((time.time() - t0) * 1000)
    ctx  = "\n\n---\n\n".join(d.page_content for d in docs)
    log.info(f"[FAISS] {len(docs)} chunks retrieved in {ms}ms ({len(ctx)} chars)")
    return ctx, ms

# ─── Startup: build / load all indexes ────────────────────────────────────────
# ── Check nomic-embed-text is available ───────────────────────────────────────
def _check_embed_model():
    """Warn clearly if the embedding model isn't pulled yet."""
    import urllib.request, urllib.error
    try:
        url = f"{OLLAMA_BASE_URL}/api/tags"
        with urllib.request.urlopen(url, timeout=3) as r:
            tags = json.loads(r.read())
        model_names = [m["name"] for m in tags.get("models", [])]
        short_names = [n.split(":")[0] for n in model_names]
        if OLLAMA_EMBED_MODEL.split(":")[0] not in short_names:
            log.warning(
                f"\n{'='*60}\n"
                f"  FAISS requires the '{OLLAMA_EMBED_MODEL}' embedding model.\n"
                f"  Run this once to enable FAISS for all agents:\n\n"
                f"      ollama pull {OLLAMA_EMBED_MODEL}\n\n"
                f"  Until then, agents will fall back to truncated injection.\n"
                f"{'='*60}"
            )
            return False
        return True
    except Exception:
        log.warning(f"Could not reach Ollama at {OLLAMA_BASE_URL} to check embedding model.")
        return False

_embed_model_ready = _check_embed_model()

log.info("Initialising knowledge bases...")

KB_RAW:   dict[str, str]            = {}
KB_INDEX: dict[str, Optional[FAISS]] = {}

for _cat in FILE_MAP:
    _raw = load_raw_kb(_cat)
    KB_RAW[_cat] = _raw
    log.info(f"  KB '{_cat}': {len(_raw):,} chars")

    if len(_raw) > FAISS_THRESHOLD_CHARS:
        if not _embed_model_ready:
            log.info(f"  KB '{_cat}': skipping FAISS (embedding model not ready)")
            KB_INDEX[_cat] = None
        else:
            try:
                KB_INDEX[_cat] = build_or_load_index(_cat, _raw)
            except Exception as e:
                log.warning(f"  FAISS unavailable for '{_cat}': {e} — using truncated injection")
                KB_INDEX[_cat] = None
    else:
        KB_INDEX[_cat] = None
        log.info(f"  KB '{_cat}': small KB — full injection, no FAISS needed")

log.info("All knowledge bases ready.")

# ─── LLM ──────────────────────────────────────────────────────────────
def get_llm(temperature: float = 0.3, num_predict: int = 800) -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        num_predict=num_predict,
    )

# ─── Base Specialist Agent ─────────────────────────────────────────────────────
class BaseSpecialistAgent:
    name:          str = "base_agent"
    kb_key:        str = ""
    system_prompt: str = ""

    def __init__(self):
        # Each agent gets its own LLM instance, FAISS index, and raw KB text
        self.llm   = get_llm(temperature=0.2)
        self.index = KB_INDEX.get(self.kb_key)
        self.raw   = KB_RAW.get(self.kb_key, "")

    # ── Public Method ──────────────────────────────────────────────────────────────
    def answer(self, session_id: str, question: str, history: list,
               original_query: str = None) -> dict:
        """
        Generate an answer to the user's question.
 
        Parameters:
            session_id     — identifies the user's conversation (for context)
            question       — the refined query from the orchestrator (used for retrieval)
            history        — recent conversation turns for follow-up question context
            original_query — the raw user message (kept for compatibility)
 
        Process:
            1. Retrieve relevant knowledge base content (via FAISS or full injection)
            2. Build the message list: system prompt + recent history + current question
            3. Send to the LLM and collect the response
            4. Extract token usage metrics for reporting
            5. Return the answer text and all metrics
 
        Returns a dict with:
            text             — the AI's answer
            response_time_ms — how long the LLM took to respond
            tokens_in/out    — token usage (input/output)
            cached           — always False (caching is disabled)
            retrieval_ms     — how long knowledge base lookup took
            chunks_used      — number of KB chunks retrieved (1 for full injection)
            retrieval_method — "faiss" or "full_injection"
        """
        # 1. Retrieval
        context, retrieval_ms, chunks_used, method = self._retrieve(question)

        # 2. Build message chain
        # history[-2:] injects the last 2 turns so follow-up questions work
        messages = [SystemMessage(content=self._build_system(context))]
        for turn in history[-2:]:
            cls = HumanMessage if turn["role"] == "user" else AIMessage
            content = turn["content"]
            if turn["role"] == "assistant" and len(content) > 300:
                content = content[:300] + "..."
            messages.append(cls(content=content))
        messages.append(HumanMessage(content=question))

        # 3. LLM call
        t0         = time.time()
        response   = self.llm.invoke(messages)
        elapsed_ms = int((time.time() - t0) * 1000)
        text       = response.content.strip()

        # 4. Tokens
        ti, to = self._extract_tokens(response, messages, text)

        log.info(
            f"[{self.name}] method={method} chunks={chunks_used} "
            f"retrieve={retrieval_ms}ms llm={elapsed_ms}ms tokens={ti}in/{to}out"
        )

        return {
            "text": text, "response_time_ms": elapsed_ms,
            "tokens_in": ti, "tokens_out": to,
            "cached": False, "retrieval_ms": retrieval_ms,
            "chunks_used": chunks_used, "retrieval_method": method,
        }

    # ── Private Helpers ─────────────────────────────────────────────────────────────
    def _retrieve(self, question: str) -> tuple:
        """
        Retrieve relevant knowledge base content for the given question.
 
        If a FAISS index exists for this agent's KB, use semantic search to
        find the top-K most relevant chunks. This is smarter and more focused.
 
        If no index exists (small KB), just take the first 8,000 characters of
        the raw text and inject them whole. Simple, but works fine for small KBs.
 
        Returns: (context_str, retrieval_ms, chunks_used, method_name)
        """
        if self.index is not None:
            ctx, ms = faiss_search(self.index, question)
            return ctx, ms, FAISS_TOP_K, "faiss"
        t0  = time.time()
        ctx = self.raw[:8_000]
        return ctx, int((time.time()-t0)*1000), 1, "full_injection"

    def _build_system(self, context: str) -> str:
        """
        Combine the agent's system prompt with the retrieved knowledge base content.
 
        The KB section is clearly marked and the model is instructed to only use
        information from it — preventing hallucinations like made-up interest rates.
        """
        kb = (
            "\n\n## Relevant Knowledge Base\n"
            "Use ONLY the information below to answer. Do not fabricate data.\n\n"
            + context
        ) if context else ""
        return self.system_prompt + kb

    @staticmethod
    def _extract_tokens(response, messages, text: str) -> tuple:
        """
        Extract input and output token counts from the LLM response.
        """
 
        ti = to = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            ti = response.usage_metadata.get("input_tokens", 0)
            to = response.usage_metadata.get("output_tokens", 0)
        if not ti and not to:
            ti = int(sum(len(m.content.split()) for m in messages) / 0.75)
            to = int(len(text.split()) / 0.75)
        return ti, to

# ─── Specialist Agents ─────────────────────────────────────────────────────────
# Each agent inherits all the logic above and just declares its name, knowledge
# base, and the system prompt that defines its personality and expertise.
class AccountsAgent(BaseSpecialistAgent):
    name   = "accounts_agent"
    kb_key = "accounts"
    system_prompt = """
You are **NAB Accounts Specialist**, a knowledgeable and warm banking assistant for NAB.

## Personality & Tone
- Professional yet approachable — like a trusted banker at a branch.
- Empathetic: acknowledge the customer's situation before diving into details.
- Clear and concise: avoid jargon; use plain English.
- Proactive: offer related tips or next steps.

## Your Expertise
- Opening a NAB bank account online (transaction and savings accounts)
- Closing or changing account details in NAB Portal Pay
- Changing preferred account names in NAB Connect
- Joint accounts and combining finances for couples
- NAB Portal Pay account administration

## Response Format
- Answer in 3–5 sentences maximum for simple questions
- Use numbered steps only when explaining a process
- Give the direct answer first, then supporting details
- Do not repeat the question back
- If the answer needs more depth, cover the key points then offer: "Would you like more detail on any of these?"
- NEVER use markdown tables (| col | col |), always use numbered lists instead

## Boundary Rule
- ONLY redirect to payments if the question is specifically about:
  sending money, international transfers, SWIFT codes, BPAY, PayID,
  transfer fees, or daily transfer limits.
- Questions about opening, closing, or managing accounts are YOUR domain.
- Questions about payee settings within internet banking are YOUR domain.
- Do NOT redirect account management questions to payments.

## Rules
1. Only answer questions related to NAB accounts.
2. Never provide specific personal financial advice.
3. If you cannot answer: "I'll connect you with a NAB human expert."
4. End with: "Is there anything else about your account I can help you with?"
5. Use numbered lists for step-by-step instructions.
"""

class BusinessAgent(BaseSpecialistAgent):
    name   = "business_agent"
    kb_key = "business"
    system_prompt = """
You are **NAB Business Banking Specialist**, an expert advisor for NAB's business banking.

## Personality & Tone
- Confident and authoritative — you understand running a business.
- Consultative, efficient, and encouraging.

## Your Expertise
## Your Expertise
- Business accounts, transaction accounts, savings and term deposits
- NAB Bookkeeper AI and business tools
- EFTPOS terminals and FlexiPurchase
- Commercial Cards Self-Service (CCSS)
- Enhanced Statement Files
- Business calculators and financial planning tools
- Starting a business checklist and finance broking
- Healthcare and specialised industry banking
- Growing business internationally
- Vehicle and equipment finance guidance

## Response Format
- Answer in 3–5 sentences maximum for simple questions
- Use numbered steps only when explaining a process
- Give the direct answer first, then supporting details
- Do not repeat the question back
- If the answer needs more depth, cover the key points then offer: "Would you like more detail on any of these?"
- NEVER use markdown tables (| col | col |), always use numbered lists instead

## Rules
1. Only answer questions relevant to business banking.
2. Never give specific tax or investment advice.
3. If the answer isn't in your knowledge base: "I'll connect you with a NAB business banking expert."
4. Highlight relevant NAB products naturally — don't hard-sell.
"""

class CardsInsuranceAgent(BaseSpecialistAgent):
    name   = "cards_insurance_agent"
    kb_key = "cards_insurance"
    system_prompt = """
You are **NAB Cards & Insurance Specialist**, expert in NAB credit cards, debit cards, and insurance.

## Personality & Tone
- Friendly and reassuring — card issues can be stressful; put customers at ease.
- Precise: card terms, rates, and conditions matter.
- Helpful: surface relevant benefits proactively.

## Your Expertise
- NAB business and personal credit cards
- Debit cards for teens (ages 8–13 and 14+)
- Card benefits and rewards programs (NAB Rewards, Qantas Points)
- Complimentary card insurances:
  - International and domestic travel insurance
  - Mobile phone insurance
  - Purchase protection insurance
  - Delayed flight lounge access
- Critical illness insurance
- Fraud protection, card security, and digital payments
- Interest-free periods, fees, and rates
- Lost/stolen card procedures
- Card schemes (Visa, Mastercard)

## Response Format
- Answer in 3–5 sentences maximum for simple questions
- Use numbered steps only when explaining a process
- Give the direct answer first, then supporting details
- Do not repeat the question back
- If the answer needs more depth, cover the key points then offer: "Would you like more detail on any of these?"
- NEVER use markdown tables (| col | col |), always use numbered lists instead

## Rules
1. Always state when rates or fees are subject to change.
2. Never provide specific financial advice.
3. For urgent issues: "Please call NAB on 13 22 65 or lost/stolen line 1800 033 103."
"""

class LoansAgent(BaseSpecialistAgent):
    name   = "loans_agent"
    kb_key = "loans"
    system_prompt = """
You are **NAB Home Loans & Lending Specialist**, a knowledgeable guide for NAB loans.

## Personality & Tone
- Calm, thorough, and transparent.
- Simplify complexity without being condescending.
- Always clarify loan type, LVR tier, and comparison rate.

## Your Expertise
- NAB home loan products (Base Variable, Tailored, Choice Package)
- Fixed vs variable rates, comparison rates, LVR tiers
- Owner-occupier vs investment loan rates
- Business loans, commercial loans, overdrafts, unsecured lending
- Vehicle and equipment loans, hire purchase, chattel mortgage
- Bridging loans and construction loans
- First Home Owner Grant (FHOG) basics
- Car loans and personal vehicle finance
- Financial hardship and loan support
- Loan application processes and required documents

## Response Format
- Answer in 3–5 sentences maximum for simple questions
- Use numbered steps only when explaining a process
- Give the direct answer first, then supporting details
- Do not repeat the question back
- If the answer needs more depth, cover the key points then offer: "Would you like more detail on any of these?"
- NEVER use markdown tables (| col | col |), always use numbered lists instead

## Rules
1. Always add: "Rates are indicative and subject to change. Contact NAB for your specific situation."
2. Never guarantee loan approval or make specific lending decisions.
3. Present rate tables clearly.
4. For complex scenarios offer to arrange a NAB home loan expert call.
"""

class PaymentsAgent(BaseSpecialistAgent):
    name   = "payments_agent"
    kb_key = "payments"
    system_prompt = """
You are **NAB Payments Specialist**, expert in NAB domestic and international payment services.

## Personality & Tone
- Precise and reassuring — money is moving; customers need confidence.
- Clear: use plain language for SWIFT codes, IBANs, BICs.
- Proactive: remind customers of cut-off times, limits, and fees.

## Your Expertise
- Domestic payments: Pay Anyone, BPAY, PayID
- International funds transfers (Internet Banking and branches)
- SWIFT codes, BIC codes, IBAN numbers
- Real-time exchange rates and cut-off times
- Transfer fees and daily limits
- Periodic and future-dated payments (domestic and international)
- Viewing payment history and transaction records
- Scheduled and future-dated payments
- FX deals (spot and forward) via NAB Connect
- Direct debits
- Linked account transfers
- Cross-border payments
- Foreign currency account withdrawals

## Response Format
- Answer in 3–5 sentences maximum for simple questions
- Use numbered steps only when explaining a process
- Give the direct answer first, then supporting details
- Do not repeat the question back
- If the answer needs more depth, cover the key points then offer: "Would you like more detail on any of these?"
- NEVER use markdown tables (| col | col |), always use numbered lists instead

## Rules
1. Always mention relevant fees and daily limits.
2. For urgent issues: "Please call NAB on 13 22 65 to initiate a trace."
3. Never ask for account numbers, PINs, or passwords.
"""

class SupportAgent(BaseSpecialistAgent):
    name   = "support_agent"
    kb_key = "support"
    system_prompt = """
You are **NAB Customer Support Specialist**, the general help desk for NAB banking services.

## Personality & Tone
- Warm, patient, and empathetic — acknowledge frustration first.
- Methodical and resourceful: guide step by step.

## Your Expertise
- NAB Portal Pay help (agents, tenants, property managers)
- Password resets and administrator access (NAB Connect)
- EFTPOS terminal setup and enhanced functions
- Cash Exchange Machines and self-service banking
- Financial hardship assistance
- Trade Finance Online profile management
- General NAB Internet Banking help
- NAB branch and contact information
- Escalation paths for unresolved issues

## Response Format
- Answer in 3–5 sentences maximum for simple questions
- Use numbered steps only when explaining a process
- Give the direct answer first, then supporting details
- Do not repeat the question back
- If the answer needs more depth, cover the key points then offer: "Would you like more detail on any of these?"
- NEVER use markdown tables (| col | col |), always use numbered lists instead

## Rules
1. Acknowledge frustration before moving to solutions.
2. Provide NAB contact details: Phone 13 22 65, Portal Pay 13 59 77.
3. Clearly explain escalation paths.
4. Confirm the customer has everything they need before closing.
"""

# ─── Orchestrator Agent ────────────────────────────────────────────────────────
#
# The orchestrator is the "front door" of the system. It reads every user
# message and decides which specialist agent should handle it. It does NOT
# answer banking questions itself — its only job is to route.
#
# It works by sending the user's message to the LLM with instructions to
# output a small JSON object like:
#   {"route": "loans_agent", "refined_query": "What home loans does NAB offer?"}
#
# That JSON is parsed and used to call the right specialist agent.
ORCHESTRATOR_SYSTEM = """
You are **NAB Virtual Assistant Orchestrator** — the intelligent front-door of NAB's AI banking help system.

## Your Role (CRITICAL)
Your ONLY job is to:
1. Greet the customer warmly and understand their query.
2. Ask ONE clarifying question ONLY if the topic is genuinely ambiguous.
3. Once confident, output a JSON routing decision.
4. NEVER answer banking questions yourself. ONLY route.

## Personality
- Warm, professional, efficient — like a knowledgeable bank receptionist.
- Patient, natural — don't make customers feel interrogated.
- Respond in Australian English ("cheers", "no worries").

## Available Agents
- **accounts_agent**        — Opening/closing bank accounts, joint accounts, combining finances, NAB Portal Pay account management
- **business_agent**        — Business banking, NAB Bookkeeper, NAB Connect, business accounts, business savings, EFTPOS, business tools
- **cards_insurance_agent** — Credit cards, debit cards, teen debit cards, rewards, fraud, travel insurance, lounge access, card insurance
- **loans_agent**           — ALL loan types: home loans, business loans, vehicle/equipment loans, car loans, bridging loans, overdrafts, financial hardship, FHOG, LVR, fixed/variable rates
- **payments_agent**        — International AND domestic transfers, SWIFT/BIC/IBAN, Pay Anyone, BPAY, PayID, FX deals, direct debits, payment limits, transaction history
- **support_agent**         — Portal Pay support, PUID, password resets, EFTPOS help, general help, login issues, branch info, contact numbers

## Routing Rules
- Route immediately when ≥70% confident. When in doubt — route, never clarify.
- ONLY ask ONE clarifying question if you genuinely cannot determine ANY agent.
- Any question mentioning a banking product name → route immediately, no clarification.
- Unknown topics → route to support_agent.

## Output Format (when routing)
Output EXACTLY this JSON, nothing else after it:
```json
{"route": "<agent_name>", "refined_query": "<clear request>"}
```

## Rules
- NEVER answer the question yourself.
- NEVER ask multiple questions at once.
- NEVER try to answer a question. ONLY route.

## Examples
"what home loans does NAB offer" → {"route": "loans_agent", "refined_query": "What home loans does NAB offer?"}
"i have an issue" → Ask: "No worries! Is it related to your account, a payment, a card, or something else?"
"i want to send money overseas" → {"route": "payments_agent", "refined_query": "How do I send money overseas with NAB?"}
"how can i make a domestic payment" → {"route": "payments_agent", "refined_query": "How do I make a domestic payment with NAB?"}
"how do i pay someone" → {"route": "payments_agent", "refined_query": "How do I pay someone using NAB?"}
"i want to know about credit cards" → {"route": "cards_insurance_agent", "refined_query": "Tell me about NAB credit cards"}
"joint account" → {"route": "accounts_agent", "refined_query": "How do I open a joint account with NAB?"}
"open a bank account" → {"route": "accounts_agent", "refined_query": "How do I open a NAB bank account online?"}
"equipment finance" → {"route": "loans_agent", "refined_query": "What equipment financing does NAB offer?"}
"create payid for business" → {"route": "payments_agent", "refined_query": "How do I create a PayID for my business?"}
"business savings account" → {"route": "business_agent", "refined_query": "What business savings accounts does NAB offer?"}
"lounge access nab card" → {"route": "cards_insurance_agent", "refined_query": "How do I get lounge access with my NAB card?"}
"teen debit card" → {"route": "cards_insurance_agent", "refined_query": "What debit cards does NAB offer for teenagers?"}
"""

class OrchestratorAgent:
    """
    Responsibilities:
      1. Receive every user message.
      2. Handle simple greetings directly (no need to involve a specialist).
      3. Ask the LLM to decide which specialist agent should answer.
      4. Parse the LLM's JSON routing decision.
      5. Fall back to keyword matching if the LLM doesn't output valid JSON.
      6. Delegate to the chosen specialist and return its answer.
    """
    def __init__(self):
        self.llm    = get_llm(temperature=0.1, num_predict=60)
        self.agents = {
            "accounts_agent":        AccountsAgent(),
            "business_agent":        BusinessAgent(),
            "cards_insurance_agent": CardsInsuranceAgent(),
            "loans_agent":           LoansAgent(),
            "payments_agent":        PaymentsAgent(),
            "support_agent":         SupportAgent(),
        }

    def process(self, session_id: str, user_message: str) -> dict:
        """
        Main entry point: process one user message and return the AI's response.
 
        Flow:
          1. Load conversation history for context.
          2. Save the new user message to history.
          3. If it's just a greeting, reply directly without involving a specialist.
          4. Send the message to the orchestrator LLM to get a routing decision.
          5. Parse the routing JSON; if missing, try keyword fallback.
          6. Delegate to the matched specialist agent and return its answer.
          7. If still no route (genuine clarification question), return the
             orchestrator's text directly.
        """
        # Step 1 & 2 — Load history, then record this new message
        history = session_get(session_id)
        session_append(session_id, "user", user_message)
        
        # Step 3 — Short-circuit for greetings, no need to route these
        GREETINGS = {"hi","hello","hey","g'day","gday","howdy","hi there","hello there","hey there","good morning","good afternoon","good evening"}
        if user_message.lower().strip().rstrip("!").rstrip(".") in GREETINGS:
           reply = "Hello! Welcome to NAB's virtual banking assistant. What can I help you with today?"
           session_append(session_id, "assistant", reply)
           return {"response": reply, "agent": "orchestrator", "status": "clarifying", "metrics": {"response_time_ms": 0, "tokens_in": 0, "tokens_out": 0, "total_tokens": 0, "cached": False, "retrieval_ms": 0, "chunks_used": 0, "retrieval_method": "none"}}
        
        # Step 4 — Ask the orchestrator LLM what to do with this message.
        # We inject the last 2 history turns so it has some context (e.g. for
        # multi-turn conversations where the user says "tell me more about that").
        messages = [SystemMessage(content=ORCHESTRATOR_SYSTEM)]
        for turn in history[-2:]:
            cls = HumanMessage if turn["role"] == "user" else AIMessage
            messages.append(cls(content=turn["content"]))
        messages.append(HumanMessage(content=user_message))

        t0_orch       = time.time()
        orch_response = self.llm.invoke(messages)
        orch_ms       = int((time.time() - t0_orch) * 1000)
        orch_text     = orch_response.content.strip()
        # Step 5 — Parse the routing JSON from the LLM's output
        route_info = self._extract_route(orch_text)

        # ── Safety net: if no JSON route, check if clarifying or direct answer ──
        if not route_info:
            clean_check = self._clean_orch_text(orch_text).strip()
            is_clarifying = (
                len(clean_check) < 300 and (
                    clean_check.endswith("?") or
                    clean_check.endswith('?"') or
                    any(q in clean_check.lower() for q in [
                        "are you asking", "could you tell", "what would you",
                        "is it related", "do you mean", "which type",
                        "can you clarify", "what kind", "are you looking",
                    ])
                )
            )
            if not is_clarifying:
                # The LLM answered directly instead of routing — override it
                log.warning(
                    f"[Orchestrator] Direct answer detected — intercepting. "
                    f"Response: {clean_check[:80]}..."
                )
                route_info = self._force_route(user_message)
        # Step 6 — Delegate to the matched specialist agent
        if route_info:
            agent_name    = route_info["route"]
            refined_query = route_info.get("refined_query", user_message)
            agent         = self.agents.get(agent_name, self.agents["support_agent"])

            t0_spec  = time.time()
            result   = agent.answer(session_id, refined_query, history,
                                    original_query=user_message)
            total_ms = int((time.time() - t0_spec) * 1000)
            # Log a clean summary to the terminal for easy debugging
            print(
                f"\n{'─'*55}\n"
                f"  Agent  : {agent_name}\n"
                f"  Time   : {(result['response_time_ms'] or total_ms) / 1000:.2f}s\n"
                f"  Tokens : {result['tokens_in']} in / {result['tokens_out']} out "
                f"(total: {result['tokens_in'] + result['tokens_out']})\n"
                f"  Method : {result['retrieval_method']} | "
                f"chunks={result['chunks_used']} | "
                f"retrieve={result['retrieval_ms']}ms\n"
                f"{'─'*55}"
            )

            session_append(session_id, "assistant", result["text"])

            return {
                "response": result["text"],
                "agent":    agent_name,
                "status":   "answered",
                "metrics": {
                    "response_time_ms": result["response_time_ms"] or total_ms,
                    "tokens_in":        result["tokens_in"],
                    "tokens_out":       result["tokens_out"],
                    "total_tokens":     result["tokens_in"] + result["tokens_out"],
                    "cached":           result["cached"],
                    "retrieval_ms":     result["retrieval_ms"],
                    "chunks_used":      result["chunks_used"],
                    "retrieval_method": result["retrieval_method"],
                },
            }
        # Step 7 — No route found; this is a genuine clarifying question from
        # the orchestrator (e.g. "Is it related to your account or a payment?").
        # Return the orchestrator's text directly to the user.
        else:
            clean = self._clean_orch_text(orch_text)
            session_append(session_id, "assistant", clean)
            ti, to = BaseSpecialistAgent._extract_tokens(orch_response, messages, clean)
            return {
                "response": clean,
                "agent":    "orchestrator",
                "status":   "clarifying",
                "metrics": {
                    "response_time_ms": orch_ms,
                    "tokens_in": ti, "tokens_out": to,
                    "total_tokens": ti + to,
                    "cached": False,
                    "retrieval_ms": 0, "chunks_used": 0,
                    "retrieval_method": "none",
                },
            }

    def _extract_route(self, text: str) -> Optional[dict]:
        """
        Try to parse a routing JSON object from the orchestrator's output.
 
        The LLM is instructed to output JSON in one of these two formats:
          1. Wrapped in a code block:  ```json { ... } ```
          2. Inline:                   {"route": "...", "refined_query": "..."}
 
        Returns the parsed dict if found, or None if no valid JSON was detected.
        """
        patterns = [
            r'```json\s*(\{[^`]+\})\s*```',
            r'(\{"route":\s*"[^"]+",\s*"refined_query":\s*"[^"]+"\})',
        ]
        for pat in patterns:
            m = re.search(pat, text, re.DOTALL)
            if m:
                try:
                    d = json.loads(m.group(1))
                    if "route" in d:
                        return d
                except Exception:
                    pass
        return None

    # Keyword - agent mapping for force-routing fallback ──────────────────
    # If the LLM outputs a direct answer instead of JSON,
    # we fall back to simple keyword matching to decide which agent to use.
    # Each agent has a list of keywords, whichever agent scores highest wins.
    FORCE_ROUTE_KEYWORDS = {
        "loans_agent": [
            "loan", "mortgage", "lvr", "interest rate", "home loan", "fixed rate",
            "variable rate", "repayment", "lend", "borrow", "property finance",
            "choice package", "investment loan", "principal", "refinanc",
            "business loan", "overdraft", "quickbiz", "unsecured loan",
            "line of credit", "equipment loan", "chattel mortgage",
            "bridging loan", "hire purchase", "car loan", "first home",  
            "fhog", "commercial loan", "financial hardship",             
        ],
        "payments_agent": [
            "transfer", "overseas", "international", "swift", "iban", "bic",
            "exchange rate", "send money", "remittance", "wire", "foreign",
            "domestic payment", "pay anyone", "bpay", "payid", "daily limit",
            "payment history", "transaction history", "how do i pay",
            "fx deal", "foreign exchange", "direct debit", "linked account", 
            "urgent payment", "foreign currency", "cross border",            
        ],
        "cards_insurance_agent": [
            "credit card", "debit card", "rewards", "qantas point", "fraud",
            "card fee", "interest free", "insurance", "lost card", "stolen card",
            "lounge access", "travel insurance", "mobile phone insurance",   
            "purchase protection", "critical illness", "teen card",          
            "teen debit", "card scheme", "visa", "mastercard",               
        ],
        "accounts_agent": [
            "account", "open account", "close account", "bank account",
            "joint account", "savings account", "transaction account",
            "portal pay account", "combining finances", "shared account",
            "nab connect account", "preferred name", "account details",
            "online account", "open a bank account",
        ],
        "business_agent": [
            "business bank", "bookkeeper", "nab bookkeeper", "small business",
            "business account", "business finance", "business tool",
            "business calculator", "business savings", "term deposit",     
            "ccss", "commercial card", "enhanced statement",                
            "flexipurchase", "eftpos", "healthcare banking",               
            "faith", "international growth", "start up", "starting a business",  
            "finance broker", "lighter capital",                            
],
    }

    def _force_route(self, user_message: str) -> dict:
        """
        Keyword-based fallback router used when the LLM doesn't produce JSON.
 
        Counts how many keywords from each agent's list appear in the user's
        message and routes to the agent with the highest score.
 
        If no keywords match at all, falls back to the support agent (catch-all).
        """
        msg_lower = user_message.lower()
        scores = {}
        for agent, keywords in self.FORCE_ROUTE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in msg_lower)
            if score > 0:
                scores[agent] = score

        if scores:
            best_agent = max(scores, key=scores.get)
            log.info(
                f"[Orchestrator] Force-routed to \'{best_agent}\' "
                f"via keyword fallback (scores: {scores})"
            )
            return {"route": best_agent, "refined_query": user_message}

        # Default fallback — support agent handles anything unknown
        log.info("[Orchestrator] No keyword match — defaulting to support_agent")
        return {"route": "support_agent", "refined_query": user_message}

    def _clean_orch_text(self, text: str) -> str:
        """
        Strip any routing JSON or code blocks from the orchestrator's output.
 
        Used when we want to return the orchestrator's clarifying question to
        the user — we don't want raw JSON leaking into the chat UI.
        """
        text = re.sub(r'```json.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'\{"route".*?\}',  '', text, flags=re.DOTALL)
        return text.strip()

    def clear_session(self, session_id: str):
        """Delete all conversation history for the given session."""
        session_clear(session_id)
