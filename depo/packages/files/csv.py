import os
from .file import File

class CSV(File):

    def _is_existed(self):
        return super()._is_existed()

##----------------------------------------------------------

    def _set_file(self):
        file_handled = os.listdir(self._get_dir())
        super()._set_file(file_handled)

##----------------------------------------------------------

    def _open_file(self):
        return open(self._get_url(), self._get_mode())

##----------------------------------------------------------

    def __open__(self):
        return super().__open__()

##----------------------------------------------------------

    def __close__(self):
        return super().__close__()