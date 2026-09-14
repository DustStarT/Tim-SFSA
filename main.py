"""Compatibility entry point for the revised Tim-SFSA pipeline.

The former monolithic entry point implemented sliding windows, fixed 48-hour
censoring and post-hoc time predictions.  Those workflows were removed for the
reviewer revision.  Invoke this file with the same arguments documented for
``run_revision_pipeline.py``.
"""

from run_revision_pipeline import main


if __name__ == "__main__":
    main()
