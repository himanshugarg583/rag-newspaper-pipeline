"""Comprehensive tests for multilingual Hindi + English newspaper support.

Covers:
1. Hindi Unicode PDF extraction.
2. English PDF extraction.
3. Hindi chunking (preserving Devanagari and sentence boundaries).
4. Hindi embedding generation (1536 dims, mocked OpenAI).
5. Hindi question against Hindi newspaper.
6. English question against Hindi newspaper (cross-lingual).
7. Hindi question against English newspaper (cross-lingual).
8. Mixed-language global retrieval (no language or document filters).
9. Hindi answer generation with page citations.
10. Garbled legacy-font PDF rejection (Kruti Dev / non-Unicode).
"""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.exceptions import CorruptedPDFTextError
from app.models.document import DocumentStatus
from app.repositories.chunk_repository import ChunkRepository, RetrievedChunk
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.language_service import DetectedLanguage, LanguageService
from app.services.llm_service import LLMService
from app.services.pdf_service import PDFService
from app.services.rag_service import RAGService


class TestMultilingualPDFExtraction:
    def test_hindi_unicode_pdf_extraction(self, tmp_path, hindi_pdf_bytes):
        """1. Test extracting authentic Devanagari Unicode Hindi text from PDF."""
        pdf_path = tmp_path / "dainik_jagran.pdf"
        pdf_path.write_bytes(hindi_pdf_bytes)

        result = PDFService().extract_pages(str(pdf_path))

        assert result.total_pages == 2
        assert len(result.pages) == 2
        assert result.pages[0].page_number == 1
        assert result.pages[1].page_number == 2
        # Verify Devanagari characters are preserved
        page_text = result.pages[0].text
        devanagari_chars = [c for c in page_text if "\u0900" <= c <= "\u097F"]
        assert len(devanagari_chars) > 20
        assert LanguageService.detect_language(page_text) == DetectedLanguage.HINDI.value

    def test_english_pdf_extraction(self, tmp_path, digital_pdf_bytes):
        """2. Test extracting English digital text from PDF."""
        pdf_path = tmp_path / "times_of_india.pdf"
        pdf_path.write_bytes(digital_pdf_bytes)

        result = PDFService().extract_pages(str(pdf_path))

        assert result.total_pages == 2
        assert len(result.pages) == 2
        assert "artificial intelligence" in result.pages[0].text.lower()
        assert LanguageService.detect_language(result.pages[0].text) == DetectedLanguage.ENGLISH.value

    def test_garbled_legacy_font_pdf_rejection(self, tmp_path, legacy_font_pdf_bytes):
        """10. Test that legacy-font encoded Hindi PDFs (e.g. Kruti Dev) are rejected."""
        pdf_path = tmp_path / "legacy_kruti.pdf"
        pdf_path.write_bytes(legacy_font_pdf_bytes)

        with pytest.raises(CorruptedPDFTextError) as exc_info:
            PDFService().extract_pages(str(pdf_path))
        assert "legacy" in exc_info.value.message.lower() or "corrupted" in exc_info.value.message.lower()


class TestMultilingualChunking:
    def test_hindi_chunking(self):
        """3. Test chunking Hindi text preserves Devanagari script, danda boundaries, and metadata."""
        service = ChunkingService(chunk_size=150, chunk_overlap=30)
        hindi_page_1 = (
            "भारतीय रिज़र्व बैंक ने आज मौद्रिक नीति समिति की बैठक में रेपो दर को 6.5 प्रतिशत पर स्थिर रखने का निर्णय लिया है। "
            "गवर्नर ने कहा कि वैश्विक अनिश्चितताओं के बीच भारतीय अर्थव्यवस्था मजबूत स्थिति में है। "
            "मुद्रास्फीति दर में क्रमिक गिरावट देखी जा रही है और विकास की गति संतोषजनक है।"
        )
        hindi_page_2 = (
            "कृषि क्षेत्र में अच्छी पैदावार से ग्रामीण मांग में सुधार की उम्मीद है। "
            "उद्योग जगत ने भी केंद्रीय बैंक के इस कदम का स्वागत किया है। "
            "शेयर बाज़ार में इस घोषणा के बाद सकारात्मक रुख देखा गया।"
        )
        pages = [(1, hindi_page_1), (2, hindi_page_2)]

        chunks = service.chunk_pages(pages)

        assert len(chunks) >= 2
        # Verify page numbers stay strictly separated
        assert any(c.page_number == 1 for c in chunks)
        assert any(c.page_number == 2 for c in chunks)
        # Verify language detected as Hindi on chunks
        assert all(c.language == DetectedLanguage.HINDI.value for c in chunks)
        # Verify sequential chunk index
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))
        # Verify Devanagari text integrity
        assert any("रेपो दर" in c.text for c in chunks)
        assert any("कृषि क्षेत्र" in c.text for c in chunks)


class TestMultilingualEmbeddings:
    def test_hindi_embedding_generation(self):
        """4. Test generating 1536-dimensional embeddings for Hindi text with mocked OpenAI."""
        client = MagicMock()
        client.embeddings.create.return_value = MagicMock(
            data=[
                MagicMock(embedding=[0.05] * 1536),
                MagicMock(embedding=[0.08] * 1536),
            ]
        )
        service = EmbeddingService(client=client)

        texts = [
            "भारत की अर्थव्यवस्था तेज़ी से बढ़ रही है।",
            "भारतीय रिज़र्व बैंक ने रेपो रेट में बदलाव नहीं किया।",
        ]
        embeddings = service.embed_texts(texts)

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1536
        assert len(embeddings[1]) == 1536
        client.embeddings.create.assert_called_once()


class TestCrossLanguageRetrievalAndQA:
    @pytest.fixture
    def hindi_chunk(self):
        return RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            filename="dainik_bhaskar.pdf",
            page_number=2,
            chunk_index=1,
            chunk_text="भारतीय रिज़र्व बैंक ने रेपो रेट को 6.5 प्रतिशत पर स्थिर रखा है। मुद्रास्फीति नियंत्रण में है।",
            distance=0.18,
        )

    @pytest.fixture
    def english_chunk(self):
        return RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            filename="economic_times.pdf",
            page_number=4,
            chunk_index=0,
            chunk_text="The Indian economy is projected to grow at 7.0 percent in the current fiscal year driven by manufacturing.",
            distance=0.19,
        )

    def test_hindi_question_against_hindi_newspaper(self, hindi_chunk):
        """5. Test answering a Hindi question using Hindi newspaper evidence."""
        mock_openai = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = (
            "भारतीय रिज़र्व बैंक ने रेपो रेट को 6.5 प्रतिशत पर स्थिर रखा है और मुद्रास्फीति नियंत्रण में है।"
        )
        mock_openai.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        service = LLMService(client=mock_openai)
        answer = service.generate_answer(
            question="आरबीआई ने रेपो रेट पर क्या फैसला लिया?",
            chunks=[hindi_chunk],
            history=[],
        )

        assert "6.5" in answer
        assert "रेपो रेट" in answer
        mock_openai.chat.completions.create.assert_called_once()

    def test_english_question_against_hindi_newspaper(self, hindi_chunk):
        """6. Test cross-language: English question answered from Hindi newspaper evidence."""
        mock_openai = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = (
            "According to the newspaper reports, the Reserve Bank of India kept the repo rate unchanged at 6.5 percent."
        )
        mock_openai.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        service = LLMService(client=mock_openai)
        answer = service.generate_answer(
            question="What is the latest decision of the central bank regarding repo rate?",
            chunks=[hindi_chunk],
            history=[],
        )

        assert "6.5" in answer
        assert "repo rate" in answer.lower()

    def test_hindi_question_against_english_newspaper(self, english_chunk):
        """7. Test cross-language: Hindi question answered in Hindi from English newspaper evidence."""
        mock_openai = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = (
            "समाचार पत्र के अनुसार, चालू वित्त वर्ष में भारतीय अर्थव्यवस्था के 7.0 प्रतिशत की दर से बढ़ने का अनुमान है।"
        )
        mock_openai.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        service = LLMService(client=mock_openai)
        answer = service.generate_answer(
            question="भारत की विकास दर के बारे में क्या अनुमान लगाया गया है?",
            chunks=[english_chunk],
            history=[],
        )

        assert "7.0" in answer
        assert "प्रतिशत" in answer or "%" in answer

    @pytest.mark.asyncio
    async def test_mixed_language_global_retrieval(self, hindi_chunk, english_chunk):
        """8. Test that global vector search queries across all completed newspapers together
        without filtering on language or document_id."""
        db = AsyncMock()
        results = MagicMock()
        # Mock database returning both Hindi and English chunks from global search
        results.all.return_value = [
            MagicMock(
                id=hindi_chunk.chunk_id,
                document_id=hindi_chunk.document_id,
                filename=hindi_chunk.filename,
                page_number=hindi_chunk.page_number,
                chunk_index=hindi_chunk.chunk_index,
                chunk_text=hindi_chunk.chunk_text,
                distance=hindi_chunk.distance,
            ),
            MagicMock(
                id=english_chunk.chunk_id,
                document_id=english_chunk.document_id,
                filename=english_chunk.filename,
                page_number=english_chunk.page_number,
                chunk_index=english_chunk.chunk_index,
                chunk_text=english_chunk.chunk_text,
                distance=english_chunk.distance,
            ),
        ]
        db.execute = AsyncMock(return_value=results)

        repository = ChunkRepository(db)
        retrieved = await repository.similarity_search_global([0.05] * 1536)

        assert len(retrieved) == 2
        assert retrieved[0].filename == "dainik_bhaskar.pdf"
        assert retrieved[1].filename == "economic_times.pdf"

        # Verify SQL statement has NO language filter and NO document_id filter
        stmt_str = str(db.execute.call_args[0][0])
        assert "language =" not in stmt_str
        assert "document_id =" not in stmt_str
        assert "documents.status" in stmt_str

    def test_hindi_answer_generation_with_page_citations(self, hindi_chunk):
        """9. Test citation metadata correctly traces Hindi chunks to their source filename and page."""
        sources = RAGService.build_sources([hindi_chunk])

        assert len(sources) == 1
        source = sources[0]
        assert source["filename"] == "dainik_bhaskar.pdf"
        assert source["page_number"] == 2
        assert source["chunk_id"] == str(hindi_chunk.chunk_id)
        assert source["document_id"] == str(hindi_chunk.document_id)
        assert source["relevance_distance"] == 0.18
        assert "रेपो रेट" in source["excerpt"]

    def test_determine_target_language(self):
        """Test accurate detection of English, Devanagari Hindi, and Roman Hindi (Hinglish)."""
        # Roman Hindi / Hinglish (Latin alphabet / abcd format)
        assert LLMService.determine_target_language("kirodi lal meena ke bare me kuch news batao") == "Hinglish"
        assert LLMService.determine_target_language("aaj ki taja khabar kya hai") == "Hinglish"
        assert LLMService.determine_target_language("modi ji ke baare me batao") == "Hinglish"
        assert LLMService.determine_target_language("jaipur me kya hua tha") == "Hinglish"
        assert LLMService.determine_target_language("bhai kuch update batao") == "Hinglish"

        # Hindi (Devanagari script)
        assert LLMService.determine_target_language("किरोड़ी लाल मीणा के बारे में कुछ बताओ") == "Hindi"
        assert LLMService.determine_target_language("जयपुर में क्या हुआ था?") == "Hindi"

        # English
        assert LLMService.determine_target_language("What is the latest news about Kirodi Lal Meena?") == "English"
        assert LLMService.determine_target_language("Tell me about the incident in Jaipur") == "English"
        assert LLMService.determine_target_language("Explain the economic growth projections") == "English"

    def test_hinglish_question_generates_roman_hindi_instruction(self, hindi_chunk):
        """Test that asking in Hinglish injects strict 26-letter Roman Hindi requirements."""
        mock_openai = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = (
            "Kirodi Lal Meena ke baare me newspaper me ye khabar aayi hai ki unhone hospital ka daura kiya."
        )
        mock_openai.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        service = LLMService(client=mock_openai)
        answer = service.generate_answer(
            question="kirodi lal meena ke bare me kuch news batao",
            chunks=[hindi_chunk],
            history=[],
        )

        assert "Kirodi Lal Meena" in answer
        call_args = mock_openai.chat.completions.create.call_args[1]
        user_msg = call_args["messages"][-1]["content"]
        assert "ROMAN HINDI / HINGLISH" in user_msg
        assert "26 standard English alphabets" in user_msg

