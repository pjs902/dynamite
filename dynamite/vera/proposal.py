"""Proposal/Result schema v1 and intake validation (spec section 4).

A Proposal is an immutable, hash-identified parameter set; a Result is the
outcome record for one proposal. Identity is environment-neutral: the
canonical JSON of the parset (sorted keys, no whitespace) hashed to 16 hex
characters, so a campaign can straddle the local box and VERA.
"""

import dataclasses
import hashlib
import json
import math
import typing

from . import SCHEMA_VERSION


def canonical_hash(parset):
    """16-hex-char identity of a parset; key-order insensitive."""
    payload = json.dumps(parset, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclasses.dataclass(frozen=True)
class Proposal:
    proposal_id: str
    parset: dict

    def to_dict(self):
        return {
            "schema_version": SCHEMA_VERSION,
            "proposal_id": self.proposal_id,
            "parset": dict(self.parset),
        }

    @staticmethod
    def from_dict(d):
        version = d.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"Proposal schema_version {version!r} != {SCHEMA_VERSION}")
        return Proposal(proposal_id=d["proposal_id"], parset=dict(d["parset"]))


@dataclasses.dataclass(frozen=True)
class Result:
    proposal_id: str
    model_dir: str
    status: str
    chi2: typing.Optional[float] = None
    kinchi2: typing.Optional[float] = None
    kinmapchi2: typing.Optional[float] = None

    def to_dict(self):
        return {"schema_version": SCHEMA_VERSION, **dataclasses.asdict(self)}

    @staticmethod
    def from_dict(d):
        version = d.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"Result schema_version {version!r} != {SCHEMA_VERSION}")
        known = {f.name for f in dataclasses.fields(Result)}
        unknown = set(d) - known - {"schema_version"}
        if unknown:
            raise ValueError(f"unknown Result field(s): {sorted(unknown)}")
        return Result(**{k: v for k, v in d.items() if k in known})


def validate_parset(parset, bounds, qobs, u_fixed=None, shape_names=None):
    """Clip parameters into bounds; reject geometrically impossible shapes.

    Returns (clipped_parset, violations). Bounds clipping silently repairs;
    triaxial feasibility violations (q > u*qobs, p < q, degenerate q) are
    hard rejections - the caller must not spend compute on them. Fixed axes
    take their parset/config values; only the free-subset case matters here.

    `shape_names` maps 'q'/'p'/'u' to the qualified parset keys of the
    STARS component (the driver builds it with Component.get_parname).
    Without it we fall back to the leading segment of each name, which is
    ambiguous: TriaxialCoredLogPotential also declares p and q, so a config
    with that halo has both p-stars and p-dh and the wrong one can win.
    """
    clipped, violations = {}, []

    def _num(x):
        return (
            isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
        )

    for name, val in parset.items():
        if not _num(val):
            violations.append(f"{name}: non-finite value {val!r}")
            continue
        lo = bounds.get(name, {}).get("lo")
        hi = bounds.get(name, {}).get("hi")
        if lo is not None and val < lo:
            val = lo
        if hi is not None and val > hi:
            val = hi
        clipped[name] = float(val)

    # Parspace names are qualified with their component -- q-stars, p-stars,
    # u-stars (config_reader builds them as f"{par}-{comp}") -- while
    # system-level names like ml are bare.
    if shape_names:
        shape = {axis: clipped[key] for axis, key in shape_names.items()
                 if key in clipped}
    else:
        shape = {}
        for name, val in clipped.items():
            shape.setdefault(name.split("-")[0], val)
    q = shape.get("q")
    p = shape.get("p")
    u = shape.get("u", u_fixed)

    if (
        q is not None
        and u is not None
        and qobs not in (None, 0)
        and q > u * qobs * (1 - 1e-6)
    ):
        violations.append(f"q={q} exceeds axis limit u*qobs={u * qobs:.4f}")
    if q is not None and p is not None and p < q:
        violations.append(f"p={p} < q={q}: oblate-equivalent shape invalid")
    if q is not None and q <= 0.05:
        violations.append(f"q={q} outside physical range")
    return clipped, violations
