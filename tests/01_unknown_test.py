from tfidf import CustomTfidfVectorizer


def test_unknown_sentence():
    input_data = ["Hello World!", "An example sentence.", "This is a test."]

    vec = CustomTfidfVectorizer(input_data)
    vec.fit()

    assert len(vec.tfidf_vectorizer.vocabulary_) == 8

    output = vec.transform(["My name is Hasan."])

    assert len(vec.tfidf_vectorizer.vocabulary_) == 11
    assert output.shape == (1, 11)
