"""
Custom exceptions for the Trip Agent application.
"""


class AppException(Exception):
    """
    Base exception class for application-specific errors.

    Attributes:
        message (str): Human-readable error message
        status_code (int): HTTP status code to return (default: 400)
        error_code (str, optional): Machine-readable error code for client handling
    """

    def __init__(self, message: str, status_code: int = 400, error_code: str = None):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(self.message)
