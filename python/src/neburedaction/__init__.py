"""Python reference implementation of the nebuRedaction secret-credential classifier.

Decides whether a single structured-log attribute holds a secret credential,
and masks it before the line is written. Standard library only.

**Experimental.** Plan §1.2 makes this release about being correct and
runnable rather than good: NR-0004 measured gate G2 as not met, and the model
card says so. Do not rely on it as a sole control.

    from neburedaction import SecretDetector, mask

    detector = SecretDetector.load("neburedaction-v0.1.0.nrm")
    result = detector.classify("ApiKey", value)
    if result.should_redact:
        value = mask(value)      # -> "***#7831"

This package and the .NET library read the same `.nrm` file and are required by
plan §6.4's G3 to agree to within 1e-6 per sample.
"""

from neburedaction.body import BodyCandidate, body_candidates
from neburedaction.detector import (
    BALANCED,
    CONSERVATIVE,
    DetectionResult,
    SecretDetector,
    SecretDetectorOptions,
    mask,
)
from neburedaction.nrm import NrmError, NrmModel, load_nrm

__all__ = [
    "BALANCED",
    "CONSERVATIVE",
    "BodyCandidate",
    "DetectionResult",
    "NrmError",
    "NrmModel",
    "SecretDetector",
    "SecretDetectorOptions",
    "__version__",
    "body_candidates",
    "has_embedded_model",
    "load_nrm",
    "mask",
]

__version__ = "0.1.0"


def has_embedded_model() -> bool:
    """Report whether this build ships a usable detection model.

    Returns:
        ``False`` for every pre-release build. It becomes ``True`` once a
        trained default model is bundled with the package (NR-0005).
    """
    return False
