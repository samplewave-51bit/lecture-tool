"""LangChain RAG engine supporting LLaMA-3 (via Groq), Google Gemini, and Local LLMs."""
import os
import logging
from typing import List, Dict, Any, Optional, Tuple, Generator
from dataclasses import dataclass

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from src.vector_store import LectureVectorStore
from src.config import RAGSettings

logger = logging.getLogger(__name__)

LECTURE_QA_SYSTEM_PROMPT = """You are an expert University Teaching Assistant and academic tutor. 
Your goal is to help students understand complex lecture material accurately, concisely, and clearly.

You are answering questions based EXCLUSIVELY on the provided lecture context from lecture slides and textbook materials.

Guidelines:
1. ACCURACY & GROUNDING: Rely strictly on the provided lecture context. Do not invent facts or extrapolate beyond what is presented in the materials. If the context does not contain sufficient information to answer the question, clearly state: "Based on the provided lecture materials, this topic is not covered in detail."
2. CITATIONS: Whenever stating a key point, definition, or formula, provide explicit source citations in square brackets, referencing the document name and slide/page number, e.g., `[Lecture 2 - Slide 7]` or `[CS101_Week1.pdf - Slide 14]`.
3. PEDAGOGICAL STRUCTURE: Format your explanation using clear bullet points, conceptual explanations, formulas in markdown/LaTeX when appropriate, and key takeaways.
4. TONE: Encouraging, academic, precise, and educational.

Context:
{context}
"""

# Known terms-gated or specialized non-chat model keywords to strictly exclude
EXCLUDED_MODEL_KEYWORDS = [
    "canopylabs", "orpheus", "whisper", "guard", "embedding", 
    "tts", "audio", "vision", "moderation", "distil", "allam"
]
ALLOWED_MODEL_FAMILIES = ["llama", "gemma", "mixtral", "qwen", "deepseek"]

# Standard verified Meta LLaMA and open models in preferred priority order
PREFERRED_LLAMA_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]

_BLACKLISTED_MODELS = set()

def fetch_groq_models(api_key: str) -> List[str]:
    """Fetch live list of chat completion models available to this Groq API key, filtering out terms-gated models."""
    if not api_key:
        return PREFERRED_LLAMA_MODELS
    try:
        from groq import Groq
        client = Groq(api_key=api_key.strip())
        resp = client.models.list()
        chat_models = []
        for m in resp.data:
            mid = m.id.lower()
            # Exclude speech, TTS, audio, terms-gated third party models (e.g. canopylabs/orpheus)
            if any(k in mid for k in EXCLUDED_MODEL_KEYWORDS):
                continue
            # Must belong to standard text LLM families
            if any(fam in mid for fam in ALLOWED_MODEL_FAMILIES):
                chat_models.append(m.id)

        # Prioritize standard LLaMA-3 models at the top
        def model_priority(m_id: str) -> int:
            if m_id in PREFERRED_LLAMA_MODELS:
                return PREFERRED_LLAMA_MODELS.index(m_id)
            elif "llama" in m_id.lower():
                return 50
            return 99

        sorted_models = sorted(chat_models, key=model_priority)
        return sorted_models if sorted_models else PREFERRED_LLAMA_MODELS
    except Exception as e:
        logger.warning(f"Error querying Groq models list: {e}")
        return PREFERRED_LLAMA_MODELS


class MockEducationalLLM:
    """Intelligent fallback for demonstration when no external API key is configured."""

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        prompt_text = messages[-1].content if messages else ""
        return AIMessage(
            content=(
                "**[Demo Mode Response - Please configure your Groq or Gemini API Key in the sidebar for full LLaMA-3 intelligence]**\n\n"
                f"**Query Received:**\n> {prompt_text}\n\n"
                "**Context Retrieved from Lecture Slides:**\n"
                "The system successfully identified relevant lecture slides and vector chunks. "
                "To generate real-time LLaMA-3 reasoning, provide a free Groq API key (`gsk_...`) or Google Gemini API key in the left sidebar.\n\n"
                "**Retrieved Sources:** See the 'Source Citations & Slide Context' expander below for exact matched lecture slides."
            )
        )


class RAGEngine:
    """Orchestrates document retrieval and contextual QA generation."""

    def __init__(self, vector_store: LectureVectorStore, settings: Optional[RAGSettings] = None):
        self.vector_store = vector_store
        self.settings = settings or RAGSettings()
        self.llm = self._init_llm()

    def update_settings(self, settings: RAGSettings):
        """Update RAG parameters and reinitialize LLM if provider/key changed."""
        self.settings = settings
        self.llm = self._init_llm()

    def _init_llm(self):
        """Initialize the selected LLM provider via LangChain."""
        provider = self.settings.llm_provider
        logger.info(f"Initializing LLM provider: {provider} with model {self.settings.llm_model}")

        # 1. Groq (LLaMA-3 models)
        if "Groq" in provider:
            groq_key = (self.settings.api_key or os.getenv("GROQ_API_KEY", "")).strip()
            if not groq_key:
                logger.warning("Groq API key not found. Using Mock fallback.")
                return MockEducationalLLM()

            # Sanitize model name: ensure it is not a terms-gated or audio/speech model
            current_model = (self.settings.llm_model or "").strip()
            if any(k in current_model.lower() for k in EXCLUDED_MODEL_KEYWORDS) or current_model in _BLACKLISTED_MODELS:
                logger.warning(f"Sanitizing gated/invalid model '{current_model}' -> 'llama-3.1-8b-instant'")
                self.settings.llm_model = "llama-3.1-8b-instant"

            try:
                from langchain_groq import ChatGroq
                return ChatGroq(
                    api_key=groq_key,
                    model_name=self.settings.llm_model,
                    temperature=self.settings.temperature,
                    max_tokens=2048
                )
            except Exception as e:
                logger.error(f"Failed to initialize ChatGroq: {e}")
                return MockEducationalLLM()

        # 2. Google Gemini
        elif "Gemini" in provider:
            gemini_key = self.settings.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
            if not gemini_key:
                logger.warning("Gemini API key not found. Using Mock fallback.")
                return MockEducationalLLM()
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    google_api_key=gemini_key,
                    model=self.settings.llm_model,
                    temperature=self.settings.temperature,
                    max_output_tokens=2048
                )
            except Exception as e:
                logger.error(f"Failed to initialize ChatGoogleGenerativeAI: {e}")
                return MockEducationalLLM()

        # 3. Ollama (Local)
        elif "Ollama" in provider:
            try:
                from langchain_community.chat_models import ChatOllama
                return ChatOllama(
                    model=self.settings.llm_model or "llama3",
                    temperature=self.settings.temperature
                )
            except Exception as e:
                logger.error(f"Failed to initialize ChatOllama: {e}")
                return MockEducationalLLM()

        return MockEducationalLLM()

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        """Invoke LLM with automatic multi-model fallback if model_terms_required or model_not_found occurs."""
        try:
            return self.llm.invoke(messages)
        except Exception as e:
            err_str = str(e)
            err_lower = err_str.lower()
            
            # Check for recoverable model errors (terms acceptance required, model not found, access denied)
            recoverable_indicators = [
                "model_terms_required",
                "terms acceptance",
                "model_not_found",
                "does not exist",
                "not have access",
                "invalid_request_error",
                "404",
                "400",
                "403"
            ]
            is_recoverable = any(ind in err_lower for ind in recoverable_indicators)

            groq_key = (self.settings.api_key or os.getenv("GROQ_API_KEY", "")).strip()
            if is_recoverable and "Groq" in self.settings.llm_provider and groq_key:
                failed_model = self.settings.llm_model
                _BLACKLISTED_MODELS.add(failed_model)
                logger.warning(f"Groq model '{failed_model}' failed ({err_str}). Blacklisting and discovering safe LLaMA candidates...")

                # Discover safe live models
                live_models = fetch_groq_models(groq_key)
                
                # Combine prioritized Meta LLaMA models with live models, excluding blacklisted and gated keywords
                candidate_pool = PREFERRED_LLAMA_MODELS + live_models
                safe_candidates = []
                for cand in candidate_pool:
                    c_low = cand.lower()
                    if cand in _BLACKLISTED_MODELS or cand in safe_candidates:
                        continue
                    if any(ex in c_low for ex in EXCLUDED_MODEL_KEYWORDS):
                        continue
                    safe_candidates.append(cand)

                for candidate in safe_candidates:
                    try:
                        logger.info(f"Attempting fallback to verified model: {candidate}")
                        from langchain_groq import ChatGroq
                        fallback_llm = ChatGroq(
                            api_key=groq_key,
                            model_name=candidate,
                            temperature=self.settings.temperature,
                            max_tokens=2048
                        )
                        response = fallback_llm.invoke(messages)
                        self.llm = fallback_llm
                        self.settings.llm_model = candidate
                        if hasattr(response, "content") and isinstance(response.content, str):
                            response.content = (
                                f"> ℹ️ *Note: Switched to verified Groq model `{candidate}` (previous model `{failed_model}` required separate terms or was unavailable).*\n\n"
                                + response.content
                            )
                        return response
                    except Exception as fb_err:
                        _BLACKLISTED_MODELS.add(candidate)
                        logger.warning(f"Fallback candidate '{candidate}' also failed: {fb_err}")

            raise e

    def format_retrieved_context(self, retrieved_docs: List[Tuple[Document, float]]) -> str:
        """Format retrieved chunks into a clean context string with citations."""
        if not retrieved_docs:
            return "No relevant lecture context found."

        context_blocks = []
        for idx, (doc, score) in enumerate(retrieved_docs, start=1):
            meta = doc.metadata
            doc_name = meta.get("doc_name", "Lecture Material")
            page_num = meta.get("page", 1)
            slide_title = meta.get("slide_title", "Slide")
            citation = meta.get("citation", f"{doc_name} (Slide {page_num})")
            
            header = f"--- [SOURCE {idx}: {citation} | Title: {slide_title} | Similarity: {score:.2f}] ---"
            block = f"{header}\n{doc.page_content.strip()}"
            context_blocks.append(block)

        return "\n\n".join(context_blocks)

    def answer_query(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        filter_doc_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Query the lecture knowledge base and generate a grounded, cited answer."""
        if self.vector_store.is_empty:
            return {
                "answer": "No lecture materials have been indexed yet. Please upload lecture PDFs or PPTs in the 'Document Studio' tab.",
                "sources": [],
                "context": ""
            }

        # 1. Retrieve relevant chunks
        retrieved_results = self.vector_store.similarity_search(
            query=query,
            k=self.settings.top_k,
            filter_doc_names=filter_doc_names
        )

        formatted_context = self.format_retrieved_context(retrieved_results)

        # 2. Build Prompt Messages
        messages: List[BaseMessage] = [
            SystemMessage(content=LECTURE_QA_SYSTEM_PROMPT.format(context=formatted_context))
        ]

        # Append recent conversation history (last 4 turns) for multi-turn context
        if chat_history:
            for turn in chat_history[-4:]:
                role = turn.get("role")
                content = turn.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))

        # Add current query
        messages.append(HumanMessage(content=query))

        # 3. Call LLM with fallback
        try:
            response = self.invoke(messages)
            answer_text = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error(f"Error during LLM invocation: {e}")
            answer_text = f"An error occurred while generating the answer: {str(e)}\n\nPlease verify your API key and network connection."

        # 4. Format structured sources
        sources_summary = []
        for doc, score in retrieved_results:
            sources_summary.append({
                "citation": doc.metadata.get("citation", doc.metadata.get("doc_name")),
                "doc_name": doc.metadata.get("doc_name"),
                "page": doc.metadata.get("page"),
                "slide_title": doc.metadata.get("slide_title"),
                "similarity": round(score, 3),
                "snippet": doc.page_content[:250] + ("..." if len(doc.page_content) > 250 else ""),
                "ocr_applied": doc.metadata.get("ocr_applied", False)
            })

        return {
            "answer": answer_text,
            "sources": sources_summary,
            "context": formatted_context
        }
