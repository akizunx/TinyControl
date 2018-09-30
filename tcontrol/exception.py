__all__ = ['WrongNumberOfArguments', 'WrongSampleTime', 'UnknownDiscretizationMethod']


class WrongNumberOfArguments(Exception):
    """
    Exception raised for not matching the numbers of parameters.
    """
    pass


class WrongSampleTime(ValueError):
    """
    Exception raised when sample time is invalid
    """
    pass


class UnknownDiscretizationMethod(ValueError):
    pass
