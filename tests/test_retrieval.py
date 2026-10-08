"""
tests/test_retrieval.py — Retrieval pipeline tests (with mocked FAISS/DB).
"""
import pytest
from unittest.mock import patch, MagicMock
import numpy as np


class TestStructuredQuery:
    def test_search_text_construction(self):
        from src.query.schema import StructuredQuery
        q = StructuredQuery(object_class="car", attributes=["red"])
        assert "red" in q.search_text
        assert "car" in q.search_text

    def test_color_hints(self):
        from src.query.schema import StructuredQuery
        q = StructuredQuery(attributes=["red", "large", "backpack"])
        assert "red" in q.color_hints
        assert "large" not in q.color_hints

    def test_attribute_normalization(self):
        from src.query.schema import StructuredQuery
        q = StructuredQuery(attributes=["RED", "Blue"])
        assert "red" in q.attributes
        assert "blue" in q.attributes

    def test_class_normalization(self):
        from src.query.schema import StructuredQuery
        q = StructuredQuery(object_class="vehicle")
        assert q.object_class == "car"

    def test_person_aliases(self):
        from src.query.schema import StructuredQuery
        q = StructuredQuery(object_class="people")
        assert q.object_class == "person"


class TestColorSimilarity:
    def test_exact_match(self):
        from src.embeddings.color import color_similarity_score
        score = color_similarity_score("red", ["red"])
        assert score == 1.0

    def test_no_match(self):
        from src.embeddings.color import color_similarity_score
        score = color_similarity_score("blue", ["red"])
        assert score == 0.0

    def test_related_colors(self):
        from src.embeddings.color import color_similarity_score
        score = color_similarity_score("orange", ["red"])
        assert score == 0.5  # related colors

    def test_none_color(self):
        from src.embeddings.color import color_similarity_score
        score = color_similarity_score(None, ["red"])
        assert score == 0.5  # neutral when unknown


class TestColorEstimation:
    def test_red_image(self):
        """Test color detection on a solid red image."""
        import numpy as np
        from src.embeddings.color import estimate_dominant_color
        # Create a solid red BGR image
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:, :, 2] = 255  # Red channel in BGR
        color = estimate_dominant_color(img)
        assert color == "red"

    def test_blue_image(self):
        """Test color detection on a solid blue image."""
        import numpy as np
        from src.embeddings.color import estimate_dominant_color
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:, :, 0] = 255  # Blue channel in BGR
        color = estimate_dominant_color(img)
        assert color == "blue"

    def test_none_on_empty(self):
        from src.embeddings.color import estimate_dominant_color
        import numpy as np
        color = estimate_dominant_color(np.array([]))
        assert color is None
