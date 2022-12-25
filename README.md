# TF-IDF-Transformation
This repository consist of a sample Python project, which follows the best practices as possible.

## Description

This package is aimed to be used for a simple TF-IDF Vectorizer. It can automatically understand whether a given sentence is trained beforehand. If it cannot find the given sentence in the vocabulary of TF-IDF, it re-calculates the TF and IDF variables and provides a proper output.

## Usage

You can use the package in two different ways:

### 1. Install from setup.py

```console
git clone https://github.com/Eruimdas/TF-IDF-Transformation/
cd TF-IDF-Transformation
pip install .
```

### 2. Install from wheel package
```console
git clone https://github.com/Eruimdas/TF-IDF-Transformation/
cd TF-IDF-Transformation
pip install dist/custom_tfidf_vectorizer-0.1-py3-none-any.whl
```

### 3. Usage Examples

You can train from scratch as follows:

```py
input_data = ["Hello World!", "An example sentence.", "This is a test."]

vec = CustomTfidfVectorizer(input_data)
vec.fit()
output = vec.transform(["This is a test."]).toarray()
vec.save("test.pkl")
```

Or you can load from a pretrained model:

```py
vec = CustomTfidfVectorizer(path="test.pkl")
output = vec.transform(["This is a test."]).toarray()
```

## Bug Reporting

Please use the following template to create a bug report:

1. Used code
2. Error output
3. Expected behaviour
