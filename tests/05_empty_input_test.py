import pytest

from tfidf import CustomTfidfVectorizer


def test_empty_input():
    with pytest.raises(ValueError, match="Either input_data or path must be provided"):
        vec = CustomTfidfVectorizer()
