from ..messages.print import Print
import errno
import os

class Error(Print):
    
    def __init__(self, type, complement):
        super()._set_type(type)
        self._set_complement(complement)

##----------------------------------------------------------

    ## Dir getter method
    def _get_complement(self):
        return self.__complement

    ## Dir setter method
    def _set_complement(self, value):
        self.__complement = value 

##----------------------------------------------------------

    def print(self):
        message = "Error: " + self._get_type().__name__ + ", " + os.strerror(errno.ENOENT) + " : " + self._get_complement()
        super()._set_message(message)
        super().print()