"""Generators for educational study aids: Summaries, FAQs, Key Concepts, and Practice Problems."""
import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document

from src.rag_engine import RAGEngine
from src.vector_store import LectureVectorStore

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = """You are an academic curriculum specialist and university professor.
Based on the provided lecture materials, generate a structured, comprehensive, and high-yield study summary.

Structure your response with the following sections:
1. 🎯 **Executive Overview**: High-level thesis of the lecture (2-3 paragraphs).
2. 📌 **Core Themes & Syllabus Breakdown**:
   - For each major theme, provide key arguments, principles, and mechanisms.
   - Include specific slide/page citations (e.g. `[Slide 5]`).
3. 🔬 **Formulas, Theorems & Key Laws** (if applicable).
4. 💡 **High-Yield Takeaways**: Bullet points highlighting critical points professors frequently test.

Lecture Context:
{context}
"""

FAQ_PROMPT = """You are an experienced teaching assistant preparing students for midterm and final exams.
Based on the provided lecture materials, generate 6 to 8 Frequently Asked Questions (FAQs) that students typically struggle with.

For each FAQ:
- **Question**: Realistic conceptual, practical, or exam-style question.
- **Answer**: Clear, pedagogical explanation grounded in the slides.
- **Reference**: Citation to relevant slide/topic (e.g., `[Slide 9: Optimization Techniques]`).

Lecture Context:
{context}
"""

KEY_CONCEPTS_PROMPT = """You are an educational designer creating a master flashcard and terminology glossary for university students.
Based on the provided lecture materials, extract and explain all Key Concepts, Terminology, Definitions, and Acronyms.

Format as a structured markdown reference table or bulleted list:
| Term / Concept | Category / Topic | Formal Definition & Intuition | Key Formula / Rule | Lecture Slide |
| --- | --- | --- | --- | --- |

Followed by a "Quick Flashcard Review" section with 4 high-priority concepts.

Lecture Context:
{context}
"""

PRACTICE_PROBLEMS_PROMPT = """You are a university professor designing an exam revision problem set.
Based on the provided lecture materials, create a comprehensive practice problem set with full solutions.

Include two distinct sections:

### Part 1: Multiple-Choice Questions (MCQs) (4 Questions)
For each question:
- State the question clearly.
- Provide 4 options (A, B, C, D).
- Specify the **Correct Answer**.
- Provide a detailed **Explanation & Reasoning** explaining why the correct option is right and common pitfalls of the distractor options.
- Citation to slide: e.g. `[Lecture 1, Slide 11]`.

### Part 2: Analytical & Conceptual Short-Answer Problems (3 Questions)
For each question:
- Problem Statement: A scenario-based, calculation, or conceptual derivation problem.
- **Hint for Student**: A nudge to guide thinking without giving away the answer.
- **Step-by-Step Solution**: Complete, rigorous solution and final answer.

Lecture Context:
{context}
"""


class EducationalContentGenerator:
    """Generates concise summaries, FAQs, key concepts, and practice problems from indexed lecture content."""

    def __init__(self, rag_engine: RAGEngine):
        self.rag_engine = rag_engine
        self.vector_store = rag_engine.vector_store

    def _collect_representative_context(
        self,
        focus_topic: str = "",
        filter_doc_names: Optional[List[str]] = None,
        max_chunks: int = 10
    ) -> str:
        """Gather representative chunks across the lecture or on a specific topic."""
        if self.vector_store.is_empty:
            return "No lecture documents have been indexed."

        queries = [focus_topic] if focus_topic else [
            "core concepts overview introduction principles",
            "definitions architecture theory methodology",
            "summary conclusions practice problems formulas"
        ]

        collected_docs: Dict[str, Document] = {}
        for q in queries:
            results = self.vector_store.similarity_search(
                query=q,
                k=max(3, max_chunks // len(queries)),
                filter_doc_names=filter_doc_names
            )
            for doc, score in results:
                chunk_id = doc.metadata.get("chunk_id", doc.page_content[:50])
                if chunk_id not in collected_docs:
                    collected_docs[chunk_id] = doc

        sorted_docs = list(collected_docs.values())[:max_chunks]
        return "\n\n".join([
            f"--- [{doc.metadata.get('citation', 'Source')} | Title: {doc.metadata.get('slide_title', '')}] ---\n{doc.page_content.strip()}"
            for doc in sorted_docs
        ])

    def generate_summary(self, focus_topic: str = "", filter_doc_names: Optional[List[str]] = None) -> str:
        """Generate concise lecture summary."""
        context = self._collect_representative_context(focus_topic, filter_doc_names, max_chunks=10)
        prompt = SUMMARY_PROMPT.format(context=context)
        response = self.rag_engine.invoke([
            SystemMessage(content="You are an expert academic curriculum designer."),
            HumanMessage(content=prompt)
        ])
        return response.content if hasattr(response, "content") else str(response)

    def generate_faqs(self, focus_topic: str = "", filter_doc_names: Optional[List[str]] = None) -> str:
        """Generate lecture FAQs."""
        context = self._collect_representative_context(focus_topic, filter_doc_names, max_chunks=10)
        prompt = FAQ_PROMPT.format(context=context)
        response = self.rag_engine.invoke([
            SystemMessage(content="You are a university exam designer and academic tutor."),
            HumanMessage(content=prompt)
        ])
        return response.content if hasattr(response, "content") else str(response)

    def generate_key_concepts(self, focus_topic: str = "", filter_doc_names: Optional[List[str]] = None) -> str:
        """Generate Key Concepts & Glossary."""
        context = self._collect_representative_context(focus_topic, filter_doc_names, max_chunks=10)
        prompt = KEY_CONCEPTS_PROMPT.format(context=context)
        response = self.rag_engine.invoke([
            SystemMessage(content="You are an educational knowledge graph engineer."),
            HumanMessage(content=prompt)
        ])
        return response.content if hasattr(response, "content") else str(response)

    def generate_practice_problems(self, focus_topic: str = "", filter_doc_names: Optional[List[str]] = None) -> str:
        """Generate exam practice problems (MCQs & analytical questions)."""
        context = self._collect_representative_context(focus_topic, filter_doc_names, max_chunks=10)
        prompt = PRACTICE_PROBLEMS_PROMPT.format(context=context)
        response = self.rag_engine.invoke([
            SystemMessage(content="You are a senior university professor writing exam questions."),
            HumanMessage(content=prompt)
        ])
        return response.content if hasattr(response, "content") else str(response)
