def check_positive(value, alias='value'):
    # Raises ValueError if the provided value is not a positive number
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(
            '{} must be a positive number, not {}'.format(alias, value)
        )


def check_positive_int(value, alias='value'):
    # Raises ValueError if the provided value is not a positive integer
    if not isinstance(value, int) or value <= 0:
        raise ValueError(
            '{} must be a positive integer, not {}'.format(alias, value)
        )
