"""Artifact-grounded model classification (spec section 5.1).

Pure filesystem logic: a model's state is read from its directory artifacts,
never from the ecsv table. Presence requires file age > min_age_s (NFS
metadata-lag guard); absence is re-checked by the caller every cycle.
"""

import enum
import os

SENTINEL = os.path.join("datfil", "tube_box_done")
WEIGHTS = "orbit_weights.ecsv"  # dyn.constants.weight_file
MIN_AGE_S_DEFAULT = 60.0  # NFS metadata-lag guard
ATTEMPT_LIMIT = 3


def _noml(model_dir):
    """Strip the trailing /mlzz.zz/ segment: orbit libraries are shared
    across ml variants and live one level up (Model.directory_noml)."""
    import re

    return re.sub(r"/ml[^/]+/?$", "/", str(model_dir))


class ModelState(enum.Enum):
    PENDING_INTEGRATION = "pending_integration"
    INTEGRATING = "integrating"
    TO_SOLVE = "to_solve"
    SOLVED = "solved"
    FAILED = "failed"  # intake rejection only; set by caller
    PARKED = "parked"


def _age(path, now_ts):
    try:
        return now_ts - os.stat(path).st_mtime
    except OSError:
        return None


def classify(model_dir, attempts, now_ts, min_age_s=MIN_AGE_S_DEFAULT):
    """A file counts as *fresh* only when its age lies in [0, min_age_s].

    Negative ages (mtime ahead of now_ts - a test clock, or NFS skew)
    carry no freshness information and are ignored.
    """
    sent = os.path.join(_noml(model_dir), SENTINEL)
    wght = os.path.join(model_dir, WEIGHTS)

    # weights presence alone means solved: the file is never transient
    if os.path.isfile(wght):
        return ModelState.SOLVED

    if attempts >= ATTEMPT_LIMIT:
        return ModelState.PARKED

    def _settled(path):
        age = _age(path, now_ts)
        return age if age is not None and age >= 0 else None

    sent_age = _settled(sent)
    if sent_age is not None:
        return ModelState.TO_SOLVE if sent_age > min_age_s else ModelState.INTEGRATING

    for root, _dirs, files in os.walk(model_dir):
        for f in files:
            age = _age(os.path.join(root, f), now_ts)
            if age is not None and 0 <= age <= min_age_s:
                return ModelState.INTEGRATING
    return ModelState.PENDING_INTEGRATION
