"""End-to-end automated tests for the Lecture RAG pipeline."""
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.utils import generate_sample_materials
from src.parser import LectureParser
from src.chunker import LectureChunker
from src.vector_store import LectureVectorStore
from src.rag_engine import RAGEngine
from src.generator import EducationalContentGenerator


class TestLectureRAGPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n--- [TEST SETUP] Generating sample materials ---")
        cls.sample_files = generate_sample_materials()
        cls.parser = LectureParser(ocr_enabled=True)
        cls.chunker = LectureChunker(chunk_size=500, chunk_overlap=80)
        cls.vector_store = LectureVectorStore(embedding_model_name="all-MiniLM-L6-v2")
        cls.rag_engine = RAGEngine(cls.vector_store)
        cls.generator = EducationalContentGenerator(cls.rag_engine)

    def test_01_pdf_parsing(self):
        print("\nTesting PDF parsing with PyMuPDF...")
        pdf_docs = self.parser.parse_file(self.sample_files["pdf"], "CS301_Lecture_04_Neural_Networks.pdf")
        self.assertGreater(len(pdf_docs), 0, "PDF should contain at least 1 page")
        self.assertEqual(pdf_docs[0].metadata["doc_type"], "PDF")
        self.assertIn("Activation Functions", pdf_docs[1].page_content)
        print(f"Extracted {len(pdf_docs)} pages from PDF successfully.")

    def test_02_pptx_parsing(self):
        print("\nTesting PPTX parsing with python-pptx...")
        pptx_docs = self.parser.parse_file(self.sample_files["pptx"], "CS301_Lecture_05_Transformers_and_Attention.pptx")
        self.assertGreater(len(pptx_docs), 0, "PPTX should contain at least 1 slide")
        self.assertEqual(pptx_docs[0].metadata["doc_type"], "PPTX")
        self.assertIn("Attention", pptx_docs[2].page_content)
        print(f"Extracted {len(pptx_docs)} slides from PPTX successfully.")

    def test_03_semantic_chunking(self):
        print("\nTesting Semantic Text Chunking...")
        pdf_docs = self.parser.parse_file(self.sample_files["pdf"], "CS301_Lecture_04_Neural_Networks.pdf")
        chunks = self.chunker.chunk_documents(pdf_docs)
        self.assertGreater(len(chunks), 0)
        self.assertIn("citation", chunks[0].metadata)
        self.assertTrue(chunks[0].page_content.startswith("[PDF:"))
        print(f"Created {len(chunks)} chunks with citation prefixes.")

    def test_04_faiss_vector_indexing(self):
        print("\nTesting FAISS Vector Indexing & Embeddings...")
        pdf_docs = self.parser.parse_file(self.sample_files["pdf"], "CS301_Lecture_04_Neural_Networks.pdf")
        pptx_docs = self.parser.parse_file(self.sample_files["pptx"], "CS301_Lecture_05_Transformers_and_Attention.pptx")
        
        pdf_chunks = self.chunker.chunk_documents(pdf_docs)
        pptx_chunks = self.chunker.chunk_documents(pptx_docs)

        self.vector_store.add_documents(pdf_chunks, "CS301_Lecture_04_Neural_Networks.pdf", "PDF", len(pdf_docs))
        self.vector_store.add_documents(pptx_chunks, "CS301_Lecture_05_Transformers_and_Attention.pptx", "PPTX", len(pptx_docs))

        self.assertFalse(self.vector_store.is_empty)
        self.assertGreater(self.vector_store.total_chunks, 0)
        
        indexed_docs = self.vector_store.get_indexed_documents()
        self.assertEqual(len(indexed_docs), 2)
        print(f"FAISS index loaded with {self.vector_store.total_chunks} total chunks across 2 lectures.")

    def test_05_similarity_search_and_filtering(self):
        print("\nTesting Similarity Search & Multi-Document Filtering...")
        # Query 1: Transformer attention (should match PPTX)
        results = self.vector_store.similarity_search("scaled dot product attention formula", k=3)
        self.assertGreater(len(results), 0)
        top_doc, score = results[0]
        self.assertIn("Attention", top_doc.page_content)
        print(f"Top match score for Attention: {score:.3f} from {top_doc.metadata['citation']}")

        # Query 2: Multi-document filter: only search PDF
        pdf_only_results = self.vector_store.similarity_search(
            "non-linear activation functions",
            k=3,
            filter_doc_names=["CS301_Lecture_04_Neural_Networks.pdf"]
        )
        self.assertGreater(len(pdf_only_results), 0)
        for d, s in pdf_only_results:
            self.assertEqual(d.metadata["doc_name"], "CS301_Lecture_04_Neural_Networks.pdf")
        print("Multi-document filtering verified successfully.")

    def test_06_rag_query_answering(self):
        print("\nTesting RAG Engine Q&A...")
        qa_result = self.rag_engine.answer_query("Why is dividing by sqrt(d_k) needed in attention?")
        self.assertIn("answer", qa_result)
        self.assertGreater(len(qa_result["sources"]), 0)
        print(f"RAG Answer Generated with {len(qa_result['sources'])} sources cited.")

    def test_07_content_generation(self):
        print("\nTesting Educational Content Generators...")
        summary = self.generator.generate_summary(focus_topic="Attention Mechanism")
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 20)

        faqs = self.generator.generate_faqs(focus_topic="Activation Functions")
        self.assertIsInstance(faqs, str)
        self.assertGreater(len(faqs), 20)

        print("Educational content generators tested successfully.")


if __name__ == "__main__":
    unittest.main()
