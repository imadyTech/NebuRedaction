"""Python reference implementation of the nebuRedaction secret-credential classifier.

This package is a placeholder established by work order NR-0000 so that the
workspace, its lint rules and its test run exist before the detector does. The
inference API is implemented in NR-0005; nothing here classifies anything yet.
"""

__all__ = ["__version__", "has_embedded_model"]

__version__ = "0.1.0"


def has_embedded_model() -> bool:
    """Report whether this build ships a usable detection model.

    Returns:
        ``False`` for every pre-release build. It becomes ``True`` once a
        trained default model is bundled with the package (NR-0005).
    """
    return False
