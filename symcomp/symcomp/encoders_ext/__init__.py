"""Extension representation arms (exploratory Stage AX sweep).

One module per arm; each defines:
  KEY: str            unique snake_case arm name
  encode(op) -> list[str]   pure deterministic tokenizer (see encoders.py)
Auto-registered into symcomp.encoders.ENCODERS at import time.
"""
