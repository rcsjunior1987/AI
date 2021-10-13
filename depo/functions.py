from .packages.files.zip import Zip as zip
from .packages.files.csv import CSV as csv

from .packages.data.data_frame import Data_Frame as data_frame

from .packages.graphs.missing_no_graphs import Missing_no_graphs
from .packages.models.analytics import Analytics

from .packages.linguistics.nlp import NLP

from .packages.metaclasses.singleton import Singleton

import os

class Functions(metaclass=Singleton):

#----------------------------------------------------------

    def open_file(dir):

        url, dir, file_name, extension_dir, extension_file  = _get_file_features(dir)

        df = data_frame()

        while switch(extension_dir):
            if case(""):
                while switch(extension_file):
                    if case("csv"):
                        df._set_file_handled(_get_file_csv(dir, file_name, url))
                        break
                    print("test")
                    break    
                break
            while switch(extension_dir):
                    if case("zip"):
                        df._set_file_handled(_get_file_zip(dir, file_name, url))
                        break
                    break    
            break

        return df._read_csv()

#----------------------------------------------------------

    def print_models_scores(X_train, X_test, y_train, y_test, type=0):
        Analytics.print_models_scores(Analytics, X_train, X_test, y_train, y_test, type)

#----------------------------------------------------------

    def print_balanded_models_scores(X, y):
        Analytics.print_balanded_models_scores(Analytics, X, y)

#----------------------------------------------------------

def _get_file_zip(dir, file_name, url):
    return zip(dir, file_name, url).__open__()

#----------------------------------------------------------

def _get_file_csv(dir, file_name, url):
    return csv(dir, file_name, url).__open__()

#----------------------------------------------------------

class switch(object):
    value = None

    def __new__(class_, value):
        class_.value = value
        return True

def case(*args):
    return any((arg == switch.value for arg in args))

#----------------------------------------------------------

def _get_file_features(dir):
    head_tail = os.path.split(dir)

    url = dir
    dir = head_tail[0]
    file_name = head_tail[1]

    extension_dir = os.path.splitext(dir)[1][1:]
    extension_file = os.path.splitext(file_name)[1][1:]

    return (url
          , dir
          , file_name
          , extension_dir
          , extension_file
           )



