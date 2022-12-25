from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="custom-tfidf-vectorizer",
    version="0.1",
    author="Hasan Kemik",
    author_email="hasan.kemikk@gmail.com",
    description="A Custom TF-IDF Vectorizer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eruimdas/TF-IDF-Transformation",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "scikit-learn==0.20.0",
        "pickleshare==0.7.5",
        "numpy==1.21.6",
        "setuptools==65.5.0"
    ],
    tests_require = ["pytest==7.2.0"]
)
