"""Interactive Streamlit Lecture RAG Application."""
import os
import io
import time
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Load local environment variables if .env exists
load_dotenv()

from src.config import (
    RAGSettings,
    AVAILABLE_EMBEDDING_MODELS,
    DEFAULT_EMBEDDING_MODEL,
    SUPPORTED_LLM_PROVIDERS,
    GROQ_MODELS,
    GEMINI_MODELS
)
from src.parser import LectureParser
from src.chunker import LectureChunker
from src.vector_store import LectureVectorStore
from src.generator import EducationalContentGenerator
from src.utils import generate_sample_materials, export_study_guide_markdown, format_file_size

try:
    from src.rag_engine import RAGEngine, fetch_groq_models
except ImportError:
    import importlib
    import src.rag_engine
    importlib.reload(src.rag_engine)
    from src.rag_engine import RAGEngine, fetch_groq_models

# Set Streamlit Page Config
st.set_page_config(
    page_title="LectureLens - Academic Lecture RAG",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .citation-badge {
        display: inline-block;
        background-color: #E0E7FF;
        color: #3730A3;
        padding: 0.2rem 0.55rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .score-badge {
        display: inline-block;
        background-color: #DEF7EC;
        color: #03543F;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .stat-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = LectureVectorStore(embedding_model_name=DEFAULT_EMBEDDING_MODEL)

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine(vector_store=st.session_state.vector_store)

if "content_generator" not in st.session_state:
    st.session_state.content_generator = EducationalContentGenerator(st.session_state.rag_engine)

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "generated_study_pack" not in st.session_state:
    st.session_state.generated_study_pack = {
        "summary": "",
        "faqs": "",
        "key_concepts": "",
        "practice_problems": ""
    }


# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/graduation-cap.png", width=64)
    st.title("🎓 LectureLens RAG")
    st.caption("AI-Powered Academic Assistant for PDFs & PPTs")

    st.markdown("---")
    st.subheader("⚙️ LLM Provider & Model")

    llm_provider = st.selectbox(
        "Provider",
        SUPPORTED_LLM_PROVIDERS,
        index=0,
        help="Choose between Groq (LLaMA-3 ultra-fast free tier), Google Gemini, or Local Ollama."
    )

    if "Groq" in llm_provider:
        default_key = os.getenv("GROQ_API_KEY", "")
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=default_key,
            placeholder="gsk_...",
            help="Free API key from https://console.groq.com"
        )

        # Dynamic model discovery from user's Groq key (cached for 10 minutes)
        @st.cache_data(ttl=600, show_spinner=False)
        def _get_live_groq_models(k: str) -> list:
            return fetch_groq_models(k)

        available_models = GROQ_MODELS
        if api_key and api_key.startswith("gsk_"):
            discovered = _get_live_groq_models(api_key)
            if discovered:
                available_models = discovered
                st.caption(f"🟢 Verified: {len(discovered)} active LLaMA models available")
            else:
                st.caption("ℹ️ Using default verified Groq catalog")

        custom_model_toggle = st.checkbox("Type Custom Model Name", value=False)
        if custom_model_toggle:
            llm_model = st.text_input("Model ID", value=available_models[0] if available_models else "llama-3.1-8b-instant")
        else:
            llm_model = st.selectbox("LLaMA / Groq Model", available_models, index=0)
    elif "Gemini" in llm_provider:
        llm_model = st.selectbox("Gemini Model", GEMINI_MODELS, index=0)
        default_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=default_key,
            placeholder="AIza...",
            help="Free key from Google AI Studio (https://aistudio.google.com)"
        )
    else:
        llm_model = st.text_input("Ollama Model Name", value="llama3")
        api_key = ""

    st.markdown("---")
    st.subheader("🔍 Retrieval Parameters")

    top_k = st.slider("Top-k Relevant Chunks", min_value=2, max_value=10, value=5)
    ocr_enabled = st.checkbox("Enable OCR Fallback", value=True, help="Extracts text from scanned slides, diagrams, and image-only pages using RapidOCR/ONNX.")
    temperature = st.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    # Sync settings with RAGEngine
    settings = RAGSettings(
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        top_k=top_k,
        ocr_enabled=ocr_enabled,
        llm_provider=llm_provider,
        llm_model=llm_model,
        temperature=temperature,
        api_key=api_key
    )
    st.session_state.rag_engine.update_settings(settings)

    st.markdown("---")
    st.subheader("📚 Multi-Document Filter")
    
    indexed_docs = st.session_state.vector_store.get_indexed_documents()
    indexed_doc_names = [d["doc_name"] for d in indexed_docs]

    if indexed_doc_names:
        filter_mode = st.radio("Retrieval Scope", ["All Semester Lectures", "Select Specific Lectures"])
        if filter_mode == "Select Specific Lectures":
            selected_docs = st.multiselect("Choose lecture files", indexed_doc_names, default=indexed_doc_names[:1])
        else:
            selected_docs = None
    else:
        selected_docs = None
        st.info("No documents indexed yet.")

    st.markdown("---")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        if st.button("Load Samples", help="Loads pre-built sample PDF & PPTX lectures on Neural Networks & Transformers."):
            with st.spinner("Loading sample lecture deck..."):
                sample_files = generate_sample_materials()
                parser = LectureParser(ocr_enabled=ocr_enabled)
                chunker = LectureChunker()
                
                # Ingest PDF
                pdf_docs = parser.parse_file(sample_files["pdf"], "CS301_Lecture_04_Neural_Networks.pdf")
                pdf_chunks = chunker.chunk_documents(pdf_docs)
                st.session_state.vector_store.add_documents(pdf_chunks, "CS301_Lecture_04_Neural_Networks.pdf", "PDF", len(pdf_docs))

                # Ingest PPTX
                pptx_docs = parser.parse_file(sample_files["pptx"], "CS301_Lecture_05_Transformers_and_Attention.pptx")
                pptx_chunks = chunker.chunk_documents(pptx_docs)
                st.session_state.vector_store.add_documents(pptx_chunks, "CS301_Lecture_05_Transformers_and_Attention.pptx", "PPTX", len(pptx_docs))

                st.success("Loaded 2 Sample Lectures!")
                time.sleep(0.5)
                st.rerun()

    with col_sb2:
        if st.button("Reset Index", help="Clears vector database and indexed documents."):
            st.session_state.vector_store.clear()
            st.session_state.chat_messages = []
            st.session_state.generated_study_pack = {"summary": "", "faqs": "", "key_concepts": "", "practice_problems": ""}
            st.warning("Index cleared.")
            time.sleep(0.5)
            st.rerun()


# ---------------------------------------------------------
# Main App Header & Metrics
# ---------------------------------------------------------
st.markdown('<div class="main-header">📚 LectureLens: Academic Lecture RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Retrieval-Augmented Generation for Lecture PDFs & PPTs: Real-time Q&A, OCR extraction, and automated study aids.</div>', unsafe_allow_html=True)

# Top Metrics Row
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
total_docs_count = len(indexed_docs)
total_chunks_count = st.session_state.vector_store.total_chunks
total_pages_count = sum(d["pages"] for d in indexed_docs)
ocr_pages_count = sum(1 for d in indexed_docs if d.get("ocr_applied"))

with col_m1:
    st.metric("Indexed Lectures", total_docs_count)
with col_m2:
    st.metric("Total Slides / Pages", total_pages_count)
with col_m3:
    st.metric("Vector Chunks", total_chunks_count)
with col_m4:
    st.metric("OCR Processed Docs", ocr_pages_count)


# ---------------------------------------------------------
# Application Navigation Tabs
# ---------------------------------------------------------
tab_studio, tab_chat, tab_study, tab_search = st.tabs([
    "📁 Document Studio",
    "💬 Lecture Q&A Chat",
    "📝 Study Pack Generator",
    "🔍 Semester Search"
])


# =========================================================
# TAB 1: Document Studio
# =========================================================
with tab_studio:
    st.subheader("📤 Ingest Lecture Material (PDF & PPT/PPTX)")
    st.markdown("Upload course materials such as professor lecture slide decks, scanned handouts, syllabus outlines, or textbook chapters.")

    uploaded_files = st.file_uploader(
        "Upload Lecture Files",
        type=["pdf", "pptx", "ppt", "txt", "md"],
        accept_multiple_files=True,
        help="Drag and drop PDF or PPTX files here. OCR will run automatically on scanned slides or image-dense diagrams."
    )

    if uploaded_files:
        if st.button("🚀 Process & Index Uploaded Documents", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            parser = LectureParser(ocr_enabled=ocr_enabled)
            chunker = LectureChunker()

            for i, uploaded_file in enumerate(uploaded_files):
                filename = uploaded_file.name
                status_text.text(f"Processing ({i+1}/{len(uploaded_files)}): {filename}...")
                
                file_bytes = io.BytesIO(uploaded_file.read())
                ext = Path(filename).suffix.lower()
                doc_type = "PPTX" if "ppt" in ext else ("PDF" if "pdf" in ext else "TEXT")

                try:
                    docs = parser.parse_file(file_bytes, filename)
                    chunks = chunker.chunk_documents(docs)
                    st.session_state.vector_store.add_documents(chunks, filename, doc_type, len(docs))
                except Exception as e:
                    st.error(f"Failed to process {filename}: {e}")

                progress_bar.progress((i + 1) / len(uploaded_files))

            status_text.text("Ingestion completed successfully!")
            st.success(f"Indexed {len(uploaded_files)} lecture file(s) into FAISS!")
            time.sleep(1)
            st.rerun()

    st.markdown("---")
    st.subheader("📋 Currently Indexed Lecture Repository")
    
    if indexed_docs:
        cols_summary = st.columns([3, 1, 1, 1, 1])
        cols_summary[0].markdown("**Document Name**")
        cols_summary[1].markdown("**Format**")
        cols_summary[2].markdown("**Slides/Pages**")
        cols_summary[3].markdown("**Vector Chunks**")
        cols_summary[4].markdown("**OCR Applied**")
        st.markdown("<hr style='margin: 0.2rem 0;'/>", unsafe_allow_html=True)

        for doc_info in indexed_docs:
            c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 1, 1])
            c1.markdown(f"📄 **{doc_info['doc_name']}**")
            c2.markdown(f"`{doc_info['doc_type']}`")
            c3.markdown(f"{doc_info['pages']}")
            c4.markdown(f"{doc_info['chunks']}")
            c5.markdown("✅ Yes" if doc_info['ocr_applied'] else "⚪ Text-native")
    else:
        st.info("No lecture files have been indexed yet. Upload files above or click **'Load Samples'** in the sidebar to test immediately.")


# =========================================================
# TAB 2: Lecture Q&A Chat
# =========================================================
with tab_chat:
    st.subheader("💬 Interactive Lecture Q&A")
    st.caption("Ask contextual questions grounded in your course materials. Answers include precise slide/page references.")

    # Quick Suggestion Buttons
    st.markdown("**Suggested Quick Questions:**")
    quick_col1, quick_col2, quick_col3 = st.columns(3)
    preset_query = None
    with quick_col1:
        if st.button("💡 Why do we need non-linear activations?"):
            preset_query = "Why do we need non-linear activations in neural networks? Compare Sigmoid and ReLU."
    with quick_col2:
        if st.button("🔍 Explain Attention formula in Slide 3"):
            preset_query = "Explain the scaled dot-product attention formula and the purpose of dividing by sqrt(d_k)."
    with quick_col3:
        if st.button("📐 How does Backpropagation work?"):
            preset_query = "Explain the backpropagation chain rule and how weight updates are computed."

    # Display Chat History
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📚 Source Citations & Matched Slides ({len(msg['sources'])})"):
                    for src in msg["sources"]:
                        st.markdown(f"""
                        <div style="margin-bottom: 0.8rem; padding: 0.5rem; background: #F9FAFB; border-left: 3px solid #3B82F6; border-radius: 4px;">
                            <span class="citation-badge">📌 {src['citation']}</span>
                            <span class="score-badge">Similarity: {src['similarity']}</span>
                            {"<span style='color: orange; font-size: 0.8rem;'>[OCR]</span>" if src.get('ocr_applied') else ""}
                            <div style="font-weight: 600; margin-top: 0.3rem;">Slide Title: {src['slide_title']}</div>
                            <div style="font-size: 0.85rem; color: #374151; margin-top: 0.2rem; font-style: italic;">"{src['snippet']}"</div>
                        </div>
                        """, unsafe_allow_html=True)

    # Chat Input Box
    user_query = st.chat_input("Ask a question about the lecture slides (e.g., 'What is vanishing gradient?')...")
    active_query = preset_query or user_query

    if active_query:
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": active_query})
        with st.chat_message("user"):
            st.markdown(active_query)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching lecture vectors and formulating grounded response..."):
                result = st.session_state.rag_engine.answer_query(
                    query=active_query,
                    chat_history=st.session_state.chat_messages[:-1],
                    filter_doc_names=selected_docs
                )

                answer_text = result["answer"]
                sources = result["sources"]

                st.markdown(answer_text)

                if sources:
                    with st.expander(f"📚 Source Citations & Matched Slides ({len(sources)})"):
                        for src in sources:
                            st.markdown(f"""
                            <div style="margin-bottom: 0.8rem; padding: 0.5rem; background: #F9FAFB; border-left: 3px solid #3B82F6; border-radius: 4px;">
                                <span class="citation-badge">📌 {src['citation']}</span>
                                <span class="score-badge">Similarity: {src['similarity']}</span>
                                {"<span style='color: orange; font-size: 0.8rem;'>[OCR]</span>" if src.get('ocr_applied') else ""}
                                <div style="font-weight: 600; margin-top: 0.3rem;">Slide Title: {src['slide_title']}</div>
                                <div style="font-size: 0.85rem; color: #374151; margin-top: 0.2rem; font-style: italic;">"{src['snippet']}"</div>
                            </div>
                            """, unsafe_allow_html=True)

                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "sources": sources
                })


# =========================================================
# TAB 3: Study Pack & Educational Content Generator
# =========================================================
with tab_study:
    st.subheader("📝 Automated Educational Content & Revision Studio")
    st.markdown("Generate comprehensive summaries, FAQs, key concept cheat-sheets, and exam practice problems directly from indexed lecture material.")

    col_topic, col_scope = st.columns([2, 1])
    with col_topic:
        focus_topic = st.text_input(
            "Topic Focus (Optional)",
            placeholder="e.g. Activation Functions, Attention Mechanism, Backpropagation, or leave empty for all",
            help="Specify a focus topic or leave blank to synthesize the entire lecture."
        )
    with col_scope:
        scope_label = "Filtered Lectures" if selected_docs else "All Semester Lectures"
        st.info(f"Target Scope: **{scope_label}**")

    st.markdown("#### Choose Content Generator")
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

    with btn_col1:
        if st.button("📖 Generate Summary", use_container_width=True):
            with st.spinner("Synthesizing concise module summary..."):
                gen_summary = st.session_state.content_generator.generate_summary(
                    focus_topic=focus_topic,
                    filter_doc_names=selected_docs
                )
                st.session_state.generated_study_pack["summary"] = gen_summary
                st.success("Summary Generated!")

    with btn_col2:
        if st.button("❓ Generate FAQs", use_container_width=True):
            with st.spinner("Generating exam FAQs and answers..."):
                gen_faqs = st.session_state.content_generator.generate_faqs(
                    focus_topic=focus_topic,
                    filter_doc_names=selected_docs
                )
                st.session_state.generated_study_pack["faqs"] = gen_faqs
                st.success("FAQs Generated!")

    with btn_col3:
        if st.button("🧠 Key Concepts", use_container_width=True):
            with st.spinner("Extracting definitions & formulas..."):
                gen_concepts = st.session_state.content_generator.generate_key_concepts(
                    focus_topic=focus_topic,
                    filter_doc_names=selected_docs
                )
                st.session_state.generated_study_pack["key_concepts"] = gen_concepts
                st.success("Key Concepts Generated!")

    with btn_col4:
        if st.button("✍️ Practice Problems", use_container_width=True):
            with st.spinner("Creating MCQs & analytical problem sets..."):
                gen_probs = st.session_state.content_generator.generate_practice_problems(
                    focus_topic=focus_topic,
                    filter_doc_names=selected_docs
                )
                st.session_state.generated_study_pack["practice_problems"] = gen_probs
                st.success("Practice Problems Generated!")

    st.markdown("---")

    # Display Generated Study Pack in Sub-Tabs
    subtab_sum, subtab_faq, subtab_con, subtab_prob = st.tabs([
        "📖 Module Summary",
        "❓ Frequently Asked Questions",
        "🧠 Key Concepts & Formulas",
        "✍️ Exam Practice Problems"
    ])

    with subtab_sum:
        if st.session_state.generated_study_pack["summary"]:
            st.markdown(st.session_state.generated_study_pack["summary"])
        else:
            st.info("Click 'Generate Summary' above to generate an executive module summary.")

    with subtab_faq:
        if st.session_state.generated_study_pack["faqs"]:
            st.markdown(st.session_state.generated_study_pack["faqs"])
        else:
            st.info("Click 'Generate FAQs' above to produce high-yield exam revision questions.")

    with subtab_con:
        if st.session_state.generated_study_pack["key_concepts"]:
            st.markdown(st.session_state.generated_study_pack["key_concepts"])
        else:
            st.info("Click 'Key Concepts' above to create a master terminology glossary and flashcard list.")

    with subtab_prob:
        if st.session_state.generated_study_pack["practice_problems"]:
            st.markdown(st.session_state.generated_study_pack["practice_problems"])
        else:
            st.info("Click 'Practice Problems' above to generate multiple-choice and analytical exam questions with full solutions.")

    # Export Study Pack Option
    st.markdown("---")
    study_pack = st.session_state.generated_study_pack
    if any(study_pack.values()):
        markdown_export = export_study_guide_markdown(
            doc_names=selected_docs or indexed_doc_names,
            summary=study_pack["summary"],
            faqs=study_pack["faqs"],
            key_concepts=study_pack["key_concepts"],
            practice_problems=study_pack["practice_problems"]
        )
        st.download_button(
            label="📥 Download Master Study Guide (.md)",
            data=markdown_export,
            file_name="LectureLens_Academic_Study_Guide.md",
            mime="text/markdown",
            type="primary"
        )


# =========================================================
# TAB 4: Semester Search & Slide Explorer
# =========================================================
with tab_search:
    st.subheader("🔍 Semester-Wide Vector Search & Slide Explorer")
    st.caption("Search across all lecture slides and notes using semantic similarity. Find where specific equations, definitions, or algorithms are introduced.")

    search_query = st.text_input("Enter search phrase or concept:", placeholder="e.g. self-attention scaling factor, vanishing gradient, He initialization")
    search_k = st.slider("Number of results to display", min_value=3, max_value=15, value=6)

    if search_query:
        if st.session_state.vector_store.is_empty:
            st.warning("Vector store is empty. Please index lecture slides first.")
        else:
            with st.spinner("Executing FAISS vector search..."):
                results = st.session_state.vector_store.similarity_search(
                    query=search_query,
                    k=search_k,
                    filter_doc_names=selected_docs
                )

                if results:
                    st.success(f"Found {len(results)} matching slide chunks:")
                    for rank, (doc, score) in enumerate(results, start=1):
                        meta = doc.metadata
                        with st.container():
                            st.markdown(f"""
                            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                    <div>
                                        <span class="citation-badge">#{rank} {meta.get('doc_name')}</span>
                                        <strong>Slide / Page: {meta.get('page')}</strong> &nbsp;|&nbsp; <em>{meta.get('slide_title', '')}</em>
                                    </div>
                                    <span class="score-badge">Relevance Score: {score:.3f}</span>
                                </div>
                                <div style="background-color: white; border: 1px solid #EDF2F7; padding: 0.8rem; border-radius: 6px; font-family: monospace; font-size: 0.9rem; white-space: pre-wrap;">
{doc.page_content}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("No matching lecture slides found for the query.")
