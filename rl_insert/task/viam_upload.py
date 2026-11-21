"""Moved to model/viam_upload.py; this name stays an alias of that module (tests patch through it)."""
import sys

from model import viam_upload as _m

sys.modules[__name__] = _m
