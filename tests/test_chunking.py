"""Chunking service tests."""

from app.services.chunking_service import ChunkingService


class TestChunking:
    def test_basic_chunking(self):
        service = ChunkingService(chunk_size=100, chunk_overlap=20)
        long_text = "The economy grew steadily this quarter. " * 30
        chunks = service.chunk_pages([(1, long_text)])

        assert len(chunks) > 1
        assert all(chunk.page_number == 1 for chunk in chunks)
        assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))

    def test_page_numbers_never_mix(self):
        """Chunks must retain the exact page their text came from."""
        service = ChunkingService(chunk_size=100, chunk_overlap=10)
        page1 = "Politics news story on page one. " * 20
        page2 = "Sports results reported on page two. " * 20
        chunks = service.chunk_pages([(1, page1), (2, page2)])

        for chunk in chunks:
            if "Politics" in chunk.text:
                assert chunk.page_number == 1
            if "Sports" in chunk.text:
                assert chunk.page_number == 2

    def test_chunk_index_is_sequential_across_pages(self):
        service = ChunkingService(chunk_size=100, chunk_overlap=10)
        chunks = service.chunk_pages([(1, "alpha " * 100), (3, "beta " * 100)])
        indices = [chunk.chunk_index for chunk in chunks]
        assert indices == list(range(len(chunks)))

    def test_skips_empty_pages(self):
        service = ChunkingService(chunk_size=100, chunk_overlap=10)
        chunks = service.chunk_pages([(1, ""), (2, "   "), (3, "real content here")])
        assert all(chunk.page_number == 3 for chunk in chunks)

    def test_small_text_single_chunk(self):
        service = ChunkingService()
        chunks = service.chunk_pages([(5, "short newspaper snippet")])
        assert len(chunks) == 1
        assert chunks[0].page_number == 5
        assert chunks[0].text == "short newspaper snippet"
