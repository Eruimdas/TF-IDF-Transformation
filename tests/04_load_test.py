import numpy as np

from tfidf import CustomTfidfVectorizer


def test_load():
    vec = CustomTfidfVectorizer(path="test.pkl")
    output = vec.transform(["This is a test."]).toarray()

    assert isinstance(output, np.ndarray)
