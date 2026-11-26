"""Pure analysis helpers for dashboards and reports (no Streamlit imports).

UI pages call these to shape tables and charts. Functions take polars frames
or pydantic manifests and return frames — unit-tested like the rest of the
library (``docs/proposals/fplt-ui-design.md``).
"""
