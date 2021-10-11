from pandas.core.base import PandasObject
from ..messages.error import Error as errorMessage
from sklearn.feature_extraction.text import CountVectorizer

from .linguistic import Linguistic

import re

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def get_token(words):
    return words.split()

def get_remove_stop_words(words):
    return [word for word in words if word not in set(stopwords.words('english'))]

def get_stemme_words(words):
    return [PorterStemmer().stem(word) for word in words]

class NLP(Linguistic):

    def _get_file_handled(self):
        return super()._get_file_handled()

#----------------------------------------------------------

    def _set_file_handled(self, value):
        return super()._set_file_handled(value)

#----------------------------------------------------------

    def _get_exception(self):
        return super()._get_exception()

#----------------------------------------------------------

    def _set_exception(self, value):
        return super()._set_exception(value)

#----------------------------------------------------------

    def __get_tokens(self):
        tokens = self.__get_tokens(self)
        return tokens

    PandasObject.get_tokens = __get_tokens

#----------------------------------------------------------

    #def __get_token(self, words):
    #    print('__get_token')
    #    return words.split()

#----------------------------------------------------------

    def __get_remove_stop_words(self):
        new_text = self.__get_remove_stop_words(self)
        return new_text

    PandasObject.__get_remove_stop_words = __get_remove_stop_words

#----------------------------------------------------------

    #def __get_remove_stop_words(self, words):
    #    return [word for word in words if word not in set(stopwords.words('english'))]

#----------------------------------------------------------

    def __get_stemme_words(self):
        words = self.__get_stemme_words(self)
        return words

    PandasObject.get_stemme_words = __get_stemme_words

#----------------------------------------------------------

    #def __get_stemme_words(self, words):
    #    return [PorterStemmer().stem(word) for word in words]

#----------------------------------------------------------

    def __get_stemme_text(self):

        corpus = []

        for i in range(0, self.shape[0]):

            #dialog = re.sub(pattern='[^a-zA-Z]', repl=' ', string=str(self)[i])
            dialog = re.sub(pattern='[^a-zA-Z]', repl=' ', string=self[i])

            dialog = dialog.lower()

            # Tokenizing the dialog/script by words
            tokens = get_token(dialog)

            # Removing the stop words
            dialog_words  = get_remove_stop_words(tokens)

            # Stemming the words
            words = get_stemme_words(dialog_words)

            dialog = ' '.join(words)

             # Creating a corpus
            corpus.append(dialog)

        return corpus

    PandasObject.get_stemme_text = __get_stemme_text

    def __get_bag_of_words(self, max_features, ngram_range):
        #corpus = self.__get_stemme_text()
        cv = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
        return cv.fit_transform(self).toarray()

    PandasObject.get_bag_of_words = __get_bag_of_words