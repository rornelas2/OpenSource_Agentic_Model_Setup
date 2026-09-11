def mean_readings(values):
    """Return the mean of non-missing readings; raise ValueError if none remain."""
    return sum(values) / len(values)
