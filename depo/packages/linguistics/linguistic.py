from ..metaclasses.interfaces import Interface, methods_required

class Linguistic(metaclass=Interface):

    def __init__(self, file_handled=None, exception=None):
        self.__file_handled = file_handled
        self.__exception = exception

##----------------------------------------------------------

    ## Dir getter method
    @methods_required
    def _get_file_handled(self):
        return self.__file_handled

##----------------------------------------------------------

    ## FileName setter method
    @methods_required
    def _set_file_handled(self, value):
        self.__file_handled = value

##----------------------------------------------------------

    ## exception getter method
    @methods_required
    def _get_exception(self):
        return self.__exception

##----------------------------------------------------------

    ## exception setter method
    @methods_required
    def _set_exception(self, value):
        self.__exception = value
