from tfidf import CustomTfidfVectorizer


def test_happy_path():
    input_data = ["Hello World!", "An example sentence.", "This is a test."]

    vec = CustomTfidfVectorizer(input_data)
    vec.fit()
    output = vec.transform(["This is a test."]).toarray()

    assert output.shape == (1, 8)
