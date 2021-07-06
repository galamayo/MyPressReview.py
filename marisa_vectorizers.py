import numpy as np
import marisa_trie
import six
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class MarisaCountVectorizer(CountVectorizer):

    fixed_vocabulary = False

    # ``CountVectorizer.fit`` method calls ``fit_transform`` so
    # ``fit`` is not provided
    def fit_transform(self, raw_documents, y=None):
        X = super(MarisaCountVectorizer, self).fit_transform(raw_documents)
        X = self._freeze_vocabulary(X)
        return X

    def _freeze_vocabulary(self, X=None):
        if not self.fixed_vocabulary:
            frozen = marisa_trie.Trie(six.iterkeys(self.vocabulary_))
            if X is not None:
                X = self._reorder_features(X, self.vocabulary_, frozen)
            self.vocabulary_ = frozen
            self.fixed_vocabulary = True
            del self.stop_words_
        return X

    def _reorder_features(self, X, old_vocabulary, new_vocabulary):
        map_index = np.empty(len(old_vocabulary), dtype=np.int32)
        for term, new_val in six.iteritems(new_vocabulary):
            map_index[new_val] = old_vocabulary[term]
        return X[:, map_index]


class MarisaTfidfVectorizer(TfidfVectorizer):

    def fit_transform(self, raw_documents, y=None):
        super(MarisaTfidfVectorizer, self).fit_transform(raw_documents)
        self._freeze_vocabulary()
        return super(MarisaTfidfVectorizer, self).fit_transform(raw_documents, y)

    def fit(self, raw_documents, y=None):
        super(MarisaTfidfVectorizer, self).fit(raw_documents)
        self._freeze_vocabulary()
        return super(MarisaTfidfVectorizer, self).fit(raw_documents, y)

    def _freeze_vocabulary(self, X=None):
        if not self.fixed_vocabulary:
            self.vocabulary_ = marisa_trie.Trie(six.iterkeys(self.vocabulary_))
            self.fixed_vocabulary = True
            del self.stop_words_
