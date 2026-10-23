"""Utilities for sample data creation, metrics, and document exporting."""
import os
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import pymupdf
from pptx import Presentation
from pptx.util import Inches, Pt

from src.config import SAMPLE_DIR

def format_file_size(size_bytes: int) -> str:
    """Format bytes into readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def generate_sample_materials() -> Dict[str, Path]:
    """Generate realistic sample lecture materials (PDF and PPTX) for immediate testing."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = SAMPLE_DIR / "CS301_Lecture_04_Neural_Networks.pdf"
    pptx_path = SAMPLE_DIR / "CS301_Lecture_05_Transformers_and_Attention.pptx"

    # 1. Create Sample PDF using PyMuPDF
    if not pdf_path.exists():
        doc = pymupdf.open()
        
        # Page 1: Title & Introduction
        page1 = doc.new_page(width=595, height=842)
        p1_text = (
            "CS301: Deep Learning & Neural Computation\n\n"
            "Lecture 04: Fundamentals of Deep Feedforward Networks\n\n"
            "Instructor: Prof. Elena Vance\n"
            "Term: Fall Semester\n\n"
            "Outline of Today's Lecture:\n"
            "1. Motivation: Linear Classifiers vs Multi-Layer Perceptrons (MLP)\n"
            "2. Activation Functions: Sigmoid, Tanh, ReLU, and GELU\n"
            "3. The Backpropagation Algorithm and Chain Rule of Calculus\n"
            "4. Vanishing and Exploding Gradient Dilemmas\n"
            "5. Weight Initialization Schemes (He vs Xavier/Glorot)\n"
        )
        page1.insert_text((50, 60), p1_text, fontsize=12, fontname="helv")

        # Page 2: Activation Functions
        page2 = doc.new_page(width=595, height=842)
        p2_text = (
            "Lecture 04 - Slide 2: Non-Linear Activation Functions\n\n"
            "Why do we need non-linear activations?\n"
            "Without non-linearities, any composition of linear layers collapses into a single linear map:\n"
            "y = W_2 * (W_1 * x + b_1) + b_2 = (W_2 * W_1) * x + (W_2 * b_1 + b_2) = W' * x + b'\n\n"
            "Key Activation Functions:\n"
            "1. Sigmoid: sigma(z) = 1 / (1 + exp(-z))\n"
            "   - Squeezes output between (0, 1)\n"
            "   - Derivative: sigma'(z) = sigma(z) * (1 - sigma(z))\n"
            "   - Major Disadvantage: Vanishing gradient for |z| >> 0, not zero-centered.\n\n"
            "2. Rectified Linear Unit (ReLU): f(z) = max(0, z)\n"
            "   - Constant gradient of 1 for z > 0 (combats vanishing gradient)\n"
            "   - Computational efficiency: simple thresholding\n"
            "   - Disadvantage: 'Dying ReLU' problem when neurons never activate (gradient = 0).\n\n"
            "3. Leaky ReLU: f(z) = max(alpha * z, z) where alpha is typically 0.01\n"
        )
        page2.insert_text((50, 60), p2_text, fontsize=12, fontname="helv")

        # Page 3: Backpropagation & Optimization
        page3 = doc.new_page(width=595, height=842)
        p3_text = (
            "Lecture 04 - Slide 3: Backpropagation and Weight Updates\n\n"
            "Backpropagation computes the gradient of the scalar loss L with respect to all network weights W.\n\n"
            "Mathematical Foundation - The Chain Rule:\n"
            "dL/dW_ij = (dL/dz_j) * (dz_j/dW_ij) = delta_j * a_i\n"
            "where delta_j is the local error term at layer l, and a_i is the activation from layer l-1.\n\n"
            "Optimization with Stochastic Gradient Descent (SGD):\n"
            "W_t+1 = W_t - eta * nabla_W L(W_t)\n"
            "where eta is the learning rate.\n\n"
            "Momentum and Adam Optimizer:\n"
            "- Momentum accelerates SGD in directions of consistent curvature: v_t = gamma * v_t-1 + eta * nabla_W L\n"
            "- Adam (Adaptive Moment Estimation) computes adaptive learning rates for each parameter using 1st and 2nd raw moment estimates.\n"
        )
        page3.insert_text((50, 60), p3_text, fontsize=12, fontname="helv")

        # Page 4: Summary & Review Questions
        page4 = doc.new_page(width=595, height=842)
        p4_text = (
            "Lecture 04 - Slide 4: Key Takeaways & Exam Preparation\n\n"
            "Crucial Points for Midterm Review:\n"
            "1. Universal Approximation Theorem: A feedforward network with a single hidden layer and non-linear activation can approximate any continuous function on compact subsets of R^n.\n"
            "2. Vanishing gradients predominantly occur in deep networks when sigmoid or tanh derivatives (< 0.25) are repeatedly multiplied across layers.\n"
            "3. He Initialization initializes weights with variance Var(W) = 2/n_in, optimized specifically for ReLU activations to prevent signal decay.\n"
            "4. Practice Exercise: Show that if f(x) is linear, an L-layer deep network has no greater expressive capacity than a 1-layer perceptron.\n"
        )
        page4.insert_text((50, 60), p4_text, fontsize=12, fontname="helv")

        doc.save(str(pdf_path))
        doc.close()

    # 2. Create Sample PPTX using python-pptx
    if not pptx_path.exists():
        prs = Presentation()

        # Slide 1: Title Slide
        slide_layout_title = prs.slide_layouts[0]
        s1 = prs.slides.add_slide(slide_layout_title)
        s1.shapes.title.text = "CS301: Lecture 05 - Transformers & Self-Attention"
        s1.placeholders[1].text = "From Recurrent Architectures to the Transformer\nInstructor: Prof. Elena Vance"

        # Slide 2: Limitations of RNNs
        slide_layout_bullets = prs.slide_layouts[1]
        s2 = prs.slides.add_slide(slide_layout_bullets)
        s2.shapes.title.text = "Slide 2: Bottlenecks of RNNs & LSTMs"
        tf2 = s2.placeholders[1].text_frame
        tf2.text = "Sequential Processing Constraint:"
        p = tf2.add_paragraph()
        p.text = "- RNNs process tokens sequentially step-by-step: h_t = f(h_t-1, x_t)"
        p = tf2.add_paragraph()
        p.text = "- Sequential dependency prohibits GPU hardware parallelization during training"
        p = tf2.add_paragraph()
        p.text = "- Information bottleneck: Long-term memory degrades over long sequences (> 50-100 steps)"
        p = tf2.add_paragraph()
        p.text = "- Solution: Attention Mechanism allows direct token-to-token connections!"
        if s2.has_notes_slide:
            s2.notes_slide.notes_text_frame.text = "Emphasize that parallel training on GPUs was the primary catalyst for the Transformer revolution."

        # Slide 3: Scaled Dot-Product Attention
        s3 = prs.slides.add_slide(slide_layout_bullets)
        s3.shapes.title.text = "Slide 3: Scaled Dot-Product Attention Formula"
        tf3 = s3.placeholders[1].text_frame
        tf3.text = "Attention Formulation (Vaswani et al. 2017):"
        p = tf3.add_paragraph()
        p.text = "Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V"
        p = tf3.add_paragraph()
        p.text = "- Q (Queries): Projections of input tokens looking for information"
        p = tf3.add_paragraph()
        p.text = "- K (Keys): Projections of tokens indicating what content they hold"
        p = tf3.add_paragraph()
        p.text = "- V (Values): The actual information representation to aggregate"
        p = tf3.add_paragraph()
        p.text = "- Scaling factor 1 / sqrt(d_k): Prevents large dot-products from pushing softmax into regions with tiny vanishing gradients!"
        if s3.has_notes_slide:
            s3.notes_slide.notes_text_frame.text = "Exam favorite question: Why divide by sqrt(d_k)? Mention the variance of dot product growing proportionally to d_k."

        # Slide 4: Multi-Head Attention & Positional Encoding
        s4 = prs.slides.add_slide(slide_layout_bullets)
        s4.shapes.title.text = "Slide 4: Multi-Head Attention & Positional Encoding"
        tf4 = s4.placeholders[1].text_frame
        tf4.text = "Key Components of Transformer Architecture:"
        p = tf4.add_paragraph()
        p.text = "1. Multi-Head Attention: Projects Q, K, V into h different subspaces, allowing the model to attend to different aspects (e.g. grammar, tense, semantics) simultaneously."
        p = tf4.add_paragraph()
        p.text = "2. Positional Encoding: Because self-attention is permutation-invariant, positional sinusoidal encodings PE(pos, 2i) = sin(pos / 10000^(2i/d)) are added to token embeddings."
        p = tf4.add_paragraph()
        p.text = "3. Layer Normalization & Residual Connections: y = LayerNorm(x + Sublayer(x)) enables stable gradient flow through hundreds of layers."

        prs.save(str(pptx_path))

    return {
        "pdf": pdf_path,
        "pptx": pptx_path
    }

def export_study_guide_markdown(
    doc_names: List[str],
    summary: str = "",
    faqs: str = "",
    key_concepts: str = "",
    practice_problems: str = ""
) -> str:
    """Combine study aids into a beautifully structured Markdown document."""
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    docs_str = ", ".join(doc_names) if doc_names else "Semester Lectures"
    
    parts = [
        f"# 📚 Semester Academic Study Guide & Revision Pack",
        f"**Generated On:** {date_str} | **Source Lectures:** `{docs_str}`\n",
        "---\n",
    ]

    if summary:
        parts.extend([
            "## 📖 1. Module Summary & Syllabus Synthesis",
            summary,
            "\n---\n"
        ])
    
    if key_concepts:
        parts.extend([
            "## 🧠 2. Key Concepts, Definitions & Formulas",
            key_concepts,
            "\n---\n"
        ])

    if faqs:
        parts.extend([
            "## ❓ 3. Frequently Asked Exam Questions (FAQs)",
            faqs,
            "\n---\n"
        ])

    if practice_problems:
        parts.extend([
            "## ✍️ 4. Exam Practice Problem Set with Detailed Solutions",
            practice_problems,
            "\n---\n"
        ])

    parts.append("\n*Generated with Lecture RAG Application (LangChain + FAISS + SentenceTransformers)*")
    return "\n\n".join(parts)
