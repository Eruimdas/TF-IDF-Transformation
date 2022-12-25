from tfidf import CustomTfidfVectorizer
import numpy as np

def test_load():
    vec = CustomTfidfVectorizer(path="test.pkl")
    output = vec.transform(["This is a test."]).toarray()

    assert isinstance(output, np.ndarray)
