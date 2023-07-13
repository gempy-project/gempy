def require_pandas():
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("The pandas library is required to use this function.")
    return pd


def require_pooch():
    try:
        import pooch
    except ImportError:
        raise ImportError("The pooch library is required to use this function.")
    return pooch


def require_gempy_legacy():
    try:
        import gempy_legacy
    except ImportError:
        raise ImportError("The gempy_legacy library is required to use this function.")
    return gempy_legacy