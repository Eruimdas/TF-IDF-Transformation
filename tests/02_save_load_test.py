from tfidf import CustomTfidfVectorizer


def test_save_load():
    input_data = ["Hello World!", "An example sentence.", "This is a test."]

    vec = CustomTfidfVectorizer(input_data)
    vec.fit()
    output1 = vec.transform(["This is a test."]).toarray()
    vec.save("test.pkl")

    vec2 = CustomTfidfVectorizer(path="test.pkl")
    output2 = vec2.transform(["This is a test."]).toarray()

    assert (output1 == output2).all()
