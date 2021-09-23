from ..metaclasses.interfaces import Interface, methods_required
from ..messages.error import Error as errorMessage

class Model(metaclass=Interface):

    def __init__(self):
        pass