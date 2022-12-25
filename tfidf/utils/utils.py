import re
from typing import List

import numpy as np
from scipy.sparse.dia import dia_matrix


def partial_fit(self, input_data: List[str]):
    """Update the vocabulary and idf of the TfidfVectorizer.

    This method updates the vocabulary_ and idf_ attributes of the TfidfVectorizer with the
    provided list of strings.

    Parameters
    ----------
    X : list of str
        A list of strings to update the vocabulary_ and idf_ attributes with.
    """

    max_idx = max(self.vocabulary_.values())
    for data in input_data:
        # update vocabulary_
        if self.lowercase:
            data = data.lower()
        tokens = re.findall(self.token_pattern, data)
        for token in tokens:
            if token not in self.vocabulary_:
                max_idx += 1
                self.vocabulary_[token] = max_idx

        # update idf_
        df = (self.n_docs + self.smooth_idf) / np.exp(self.idf_ - 1) - self.smooth_idf
        self.n_docs += 1
        df.resize(len(self.vocabulary_))
        for token in tokens:
            df[self.vocabulary_[token]] += 1
        idf = np.log((self.n_docs + self.smooth_idf) / (df + self.smooth_idf)) + 1
        self._tfidf._idf_diag = dia_matrix((idf, 0), shape=(len(idf), len(idf)))
