from .file import File
from zipfile import ZipFile

class Zip(File):

    def _is_existed(self):
        return super()._is_existed()

##----------------------------------------------------------

    def _set_file(self):
        file_handled = ZipFile(self._get_dir())        
        super()._set_file(file_handled)

##----------------------------------------------------------

    def _open_file(self):
        return super()._get_file_handled().open(self._get_file_name())

##----------------------------------------------------------

    def __open__(self):
        return super().__open__()

##----------------------------------------------------------

    def __close__(self):
        return super().close()