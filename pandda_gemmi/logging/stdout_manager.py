import contextlib


class STDOUTManager(contextlib.ContextDecorator):
    def __init__(self,enter_message, exit_message):
        self.enter_message = enter_message
        self.exit_message = exit_message

    def __enter__(self, ):
        print(self.enter_message)

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(self.exit_message)