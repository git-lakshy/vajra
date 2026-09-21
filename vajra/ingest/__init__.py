"""Ingestion adapters: synthetic replay (default), MRMS grib2, SEVIR h5.

All adapters expose the same frame API:
    fields = adapter.current_frame()   -> dict[str, np.ndarray]
    adapter.advance()                  -> move to the next 5-min frame
"""
