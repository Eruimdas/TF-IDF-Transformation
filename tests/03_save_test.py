import os

from tfidf import CustomTfidfVectorizer


def test_save():
    input_data = ["Hello World!", "An example sentence.", "This is a test."]

    vec = CustomTfidfVectorizer(input_data)
    vec.fit()
    output1 = vec.transform(["This is a test."])
    vec.save("test.pkl")

    assert os.path.exists("test.pkl")
