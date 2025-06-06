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


def require_gempy_viewer():
    try:
        import gempy_viewer
    except ImportError:
        raise ImportError("The gempy_viewer package is required to run this function.")
    return gempy_viewer

def require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("The matplotlib package is required to run this function.")
    return plt


def require_skimage():
    try:
        import skimage
    except ImportError:
        raise ImportError("The skimage package is required to run this function.")
    return skimage


def require_scipy():
    try:
        import scipy
    except ImportError:
        raise ImportError("The scipy package is required to run this function.")
    return scipy


def require_subsurface():
    try:
        import subsurface
    except ImportError:
        raise ImportError("The subsurface package is required to run this function.")
    return subsurface

def require_zlib():
    try:
        import zlib
    except ImportError:
        raise ImportError("The zlib package is required to run this function.")
    return zlib

def require_torch():
    try:
        import torch
    except ImportError:
        raise ImportError("The torch package is required to run this function.")
    return torch