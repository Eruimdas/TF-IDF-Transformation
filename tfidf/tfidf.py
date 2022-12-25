from sklearn.feature_extraction.text import TfidfVectorizer
from .utils.utils import partial_fit
import pickle
import re
from typing import Union, List


class CustomTfidfVectorizer:
    """A custom TfidfVectorizer class that can fit and transform new data incrementally.

    This class wraps the TfidfVectorizer class from scikit-learn, and adds the ability
    to fit and transform new data incrementally using the partial_fit method.

    Parameters
    ----------
    input_data : list or str, optional
        A list of strings or a single string to be used for fitting the TfidfVectorizer.
        Either input_data or path must be provided.
    path : str, optional
        The path to a pickled TfidfVectorizer object. Either input_data or path must be provided.
    **kwargs : dict
        Additional keyword arguments to pass to the TfidfVectorizer constructor.

    Attributes
    ----------
    input_data : list or str
        The input data used for fitting the TfidfVectorizer.
    tfidf_vectorizer : TfidfVectorizer
        The TfidfVectorizer object used for fitting and transforming data.
    """

    def __init__(
        self,
        input_data: Union[List[str], str, None] = None,
        path: Union[str, None] = None,
        **kwargs
    ):
        """Constructor for the CustomTfidfVectorizer class.

        Either input_data or path must be provided. If input_data is provided, a new
        TfidfVectorizer object will be created and fit on the input data. If path is provided,
        the TfidfVectorizer object will be loaded from the specified path.

        Parameters
        ----------
        input_data : list or str, optional
            A list of strings or a single string to be used for fitting the TfidfVectorizer.
            Either input_data or path must be provided.
        path : str, optional
            The path to a pickled TfidfVectorizer object. Either input_data or path must be provided.
        **kwargs : dict
            Additional keyword arguments to pass to the TfidfVectorizer constructor.

        Raises
        ------
        ValueError
            If neither input_data nor path is provided.
        """
        TfidfVectorizer.partial_fit = partial_fit
        self.tfidf_vectorizer = TfidfVectorizer(**kwargs)
        if path is None and input_data is not None:
            self.input_data = input_data
        elif path is not None and input_data is None:
            self.load(path)
        else:
            raise ValueError("Either input_data or path must be provided")

    def check_input_data(self, input_data):
        if input_data is None and self.input_data is None:
            raise ValueError("input_data must be provided")
        elif input_data is None and self.input_data is not None:
            input_data = self.input_data
        return input_data
    
    def fit(self, input_data: Union[List[str], str, None] = None):
        """Fit the TfidfVectorizer on the input data.

        This method fits the TfidfVectorizer on the input data, updating the vocabulary_
        and idf_ attributes of the TfidfVectorizer object. If input_data is not provided,
        the input_data attribute of the CustomTfidfVectorizer object will be used.

        Parameters
        ----------
        input_data : list or str, optional
            A list of strings or a single string to be used for fitting the TfidfVectorizer.
            If not provided, the input_data attribute of the CustomTfidfVectorizer object will be used.

        Raises
        ------
        ValueError
            If input_data is not provided and the input_data attribute of the CustomTfidfVectorizer object is not set.
        """

        input_data = self.check_input_data(input_data)
        self.tfidf_vectorizer = self.tfidf_vectorizer.fit(input_data)
        self.tfidf_vectorizer.n_docs = len(input_data)

    def check_new_corpus(self, new_data: Union[List[str], str]):
        """Check for new words in the input data and fit the TfidfVectorizer on them incrementally.

        This method checks for new words in the input data and fits the TfidfVectorizer on them
        incrementally, updating the vocabulary_ and idf_ attributes of the TfidfVectorizer object.

        Parameters
        ----------
        new_data : list or str
            A list of strings or a single string to be checked for new words.
        """

        new_corpus = set()

        if isinstance(new_data, str):
            new_data = [new_data]

        if not isinstance(new_data, list):
            raise ValueError("new_data must be a list of strings or a single string")

        # If any of the words in the input data are not in the vocabulary, add them to the new_corpus set
        for data in new_data:
            if any(
                word.lower() not in self.tfidf_vectorizer.vocabulary_
                for word in re.findall(self.tfidf_vectorizer.token_pattern, data)
            ):
                new_corpus.add(data)

        new_corpus = list(new_corpus)
        if len(new_corpus) > 0:
            self.tfidf_vectorizer.partial_fit(new_corpus)

    def transform(self, input_data: Union[List[str], str, None] = None):
        """Transform the input data using the TfidfVectorizer.

        This method transforms the input data using the TfidfVectorizer, checking for new words
        in the input data and fitting the TfidfVectorizer on them incrementally if necessary.

        Parameters
        ----------
        input_data : list or str, optional
            A list of strings or a single string to be transformed. If not provided, the input_data
            attribute of the CustomTfidfVectorizer object will be used.

        Returns
        -------
        scipy.sparse.csr.csr_matrix
            The transformed input data as a sparse matrix.

        Raises
        ------
        ValueError
            If input_data is not provided and the input_data attribute of the CustomTfidfVectorizer object is not set.
        """

        input_data = self.check_input_data(input_data)
        self.check_new_corpus(input_data)
        transformed_data = self.tfidf_vectorizer.transform(input_data)
        return transformed_data

    def fit_transform(self, input_data: Union[List[str], str, None] = None):
        """Fit and transform the input data using the TfidfVectorizer.

        This method fits and transforms the input data using the TfidfVectorizer, updating the
        vocabulary_ and idf_ attributes of the TfidfVectorizer object. If input_data is not provided,
        the input_data attribute of the CustomTfidfVectorizer object will be used.

        Parameters
        ----------
        input_data : list or str, optional
            A list of strings or a single string to be transformed. If not provided, the input_data
            attribute of the CustomTfidfVectorizer object will be used.

        Returns
        -------
        scipy.sparse.csr.csr_matrix
            The transformed input data as a sparse matrix.

        Raises
        ------
        ValueError
            If input_data is not provided and the input_data attribute of the CustomTfidfVectorizer object is not set.
        """

        input_data = self.check_input_data(input_data)
        transformed_data = self.tfidf_vectorizer.fit_transform(input_data)
        return transformed_data

    def save(self, file_name: str):
        """Save the TfidfVectorizer object to a pickle file.

        This method saves the TfidfVectorizer object to a pickle file with the specified file name.

        Parameters
        ----------
        file_name : str
            The file name to use for the pickle file.
        """

        with open(file_name, "wb") as file:
            pickle.dump(self.tfidf_vectorizer, file)

    def load(self, file_name: str):
        """Load the TfidfVectorizer object from a pickle file.

        This method loads the TfidfVectorizer object from a pickle file with the specified file name.

        Parameters
        ----------
        file_name : str
            The file name of the pickle file to load.
        """

        with open(file_name, "rb") as file:
            self.tfidf_vectorizer = pickle.load(file)
