import sys
import traceback
from typing import Optional


class DocumentPortalException(Exception):

    def __init__(
        self,
        error_message,
        error_details: Optional[object] = None
    ):

        # Normalize error message
        self.error_message = str(error_message)

        exc_type = None
        exc_value = None
        exc_tb = None

        # Case 1: Exception object explicitly passed
        if isinstance(error_details, BaseException):
            exc_type = type(error_details)
            exc_value = error_details
            exc_tb = error_details.__traceback__

        # Case 2: sys module passed (old usage pattern)
        elif error_details is sys:
            exc_type, exc_value, exc_tb = sys.exc_info()

        # Case 3: Use current exception context
        elif error_details is None:
            exc_type, exc_value, exc_tb = sys.exc_info()

        # Fallback
        else:
            exc_type, exc_value, exc_tb = sys.exc_info()

        # ----------------------------------------
        # Find deepest traceback frame
        # ----------------------------------------

        last_tb = exc_tb

        while last_tb and last_tb.tb_next:
            last_tb = last_tb.tb_next

        if last_tb:
            self.file_name = last_tb.tb_frame.f_code.co_filename
            self.lineno = last_tb.tb_lineno
        else:
            self.file_name = "<unknown>"
            self.lineno = -1

        # ----------------------------------------
        # Full traceback
        # ----------------------------------------

        if exc_type and exc_tb:
            self.traceback_str = "".join(
                traceback.format_exception(
                    exc_type,
                    exc_value,
                    exc_tb
                )
            )
        else:
            self.traceback_str = ""

        super().__init__(self.error_message)

    def __str__(self):

        message = (
            f"Error in [{self.file_name}] "
            f"at line [{self.lineno}] | "
            f"Message: {self.error_message}"
        )

        if self.traceback_str:
            return (
                f"{message}\n"
                f"Traceback:\n"
                f"{self.traceback_str}"
            )

        return message

    def __repr__(self):

        return (
            f"DocumentPortalException("
            f"file={self.file_name!r}, "
            f"line={self.lineno}, "
            f"message={self.error_message!r})"
        )