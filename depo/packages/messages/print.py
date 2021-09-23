from ..metaclasses.interfaces import Interface, methods_required

import logging

class Print(metaclass=Interface):

    def __init__(self, message, type):
        self.__message = message
        self.__type = type

##----------------------------------------------------------

    ## Dir getter method
    def _get_message(self):
        return self.__message

    ## Dir setter method
    def _set_message(self, value):
        self.__message = value        

##----------------------------------------------------------

    ## Dir getter method
    def _get_type(self):
        return self.__type

    ## Dir setter method
    def _set_type(self, value):
        self.__type = value 

##----------------------------------------------------------

    @methods_required
    def print(self):
        logging.error(self._get_message())
        #print(self._get_message())