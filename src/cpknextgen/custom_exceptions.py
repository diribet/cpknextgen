class InvalidDataError(Exception):
    def __init__(self, message: str):
        self.message = message


class InvalidParameterError(Exception):
    def __init__(self, message: str):
        self.message = message
