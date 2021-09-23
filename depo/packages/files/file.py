from ..metaclasses.interfaces import Interface, methods_required
from ..messages.error import Error as errorMessage

class File(metaclass=Interface):

    def __init__(self, dir=None, file_name=None, url=None, file_handled=None, exception=None, mode="r"):
        self.__dir = dir
        self.__file_name = file_name
        self.__url = url
        self.__mode = mode
        self.__file_handled = file_handled
        self.__exception = exception

##----------------------------------------------------------

    ## Url getter method
    def _get_url(self):
        return self.__url

##----------------------------------------------------------

    ## Url setter method
    def _set_url(self, value):
        self.__url = value

##----------------------------------------------------------

    ## Dir getter method
    def _get_dir(self):
        return self.__dir

##----------------------------------------------------------

    ## Dir setter method
    def _set_dir(self, value):
        self.__dir = value

##----------------------------------------------------------

    ## FileName getter method
    def _get_file_name(self):
        return self.__file_name

##----------------------------------------------------------

    ## FileName setter method
    def _set_file_name(self, value):
        self.__file_name = value

##----------------------------------------------------------

    ## Mode getter method
    def _get_mode(self):
        return self.__mode

##----------------------------------------------------------

    ## FileName setter method
    def _set_mode(self, value):
        self.__mode = value

##----------------------------------------------------------

    ## Dir getter method
    def _get_file_handled(self):
        return self.__file_handled

##----------------------------------------------------------

    ## FileName setter method
    def _set_file_handled(self, value):
        self.__file_handled = value

##----------------------------------------------------------

    ## exception getter method
    def _get_exception(self):
        return self.__exception

##----------------------------------------------------------

    ## exception setter method
    def _set_exception(self, value):
        self.__exception = value

##----------------------------------------------------------

    @methods_required
    def _set_file(self, value):
        self._set_file_handled(value)

##----------------------------------------------------------

    @methods_required
    def _open_file(self):
        pass

##----------------------------------------------------------

    @methods_required
    def _is_existed(self):
        pass

##----------------------------------------------------------

    @methods_required
    def __open__(self):

        self._set_exception(None)

        try:
            self._set_file()

            if (self._get_file_handled() != None):
                self._set_file_handled(self._open_file())

                return self._get_file_handled()
        except Exception as e:
            self._set_exception(e.__class__)
            errorMessage(self._get_exception(), self._get_url()).print()     
                
##----------------------------------------------------------

    @methods_required    
    def __close__(self):
        pass


       