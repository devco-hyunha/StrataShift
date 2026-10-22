"""Find missing prerequisites and install them on demand (p6-12, Windows stage 3d).

I/O only — kept apart from conversion/quantization. Modelled on the pinned engine's
setup.py (pin c566ff4, read-only, never imported): probe → plan → ask once → install →
probe again.
"""
