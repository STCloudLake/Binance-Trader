"""Feature contract v1: the frozen hash must be **recomputable** (audit D-4).

The research reference (``docs/research/CORE_ALGORITHMS.md`` §11 D-4) found that
``core/ml/features.py`` declared ``FEATURE_SCHEMA_V1_HASH = "335e63360104"`` while
its own comment said the header listed 39 features and the list actually held 40
(bare ``hurst`` was still there at that revision).  Independent recomputation
showed the declared hash *was* the 39-column list and the 40-column list hashed
to ``70899cff156d`` — i.e. the literal was right and nothing in the module could
prove it.

The fix: the v1 contract is now **derived** (``DEFAULT_FEATURES`` minus the named
P6-B ``VOLUME_FLOW_FEATURES`` family) and its hash is computed from that list, so
the two cannot drift.  These tests recompute it, pin the value every archived v1
model carries, and re-derive the research document's two numbers.
"""
from __future__ import annotations

import hashlib
import json

import pytest


def test_the_v1_contract_is_the_v2_list_minus_the_p6b_family():
    from core.ml.features import (DEFAULT_FEATURES, FEATURE_NAMES,
                                  FEATURE_V1_NAMES, VOLUME_FLOW_FEATURES,
                                  FEATURE_SCHEMA_VERSION)
    assert FEATURE_SCHEMA_VERSION == 2
    assert len(DEFAULT_FEATURES) == len(FEATURE_NAMES) == 54
    assert len(VOLUME_FLOW_FEATURES) == 15
    # Order matters: the v1 list is the v2 list with the family removed, so the
    # v1 hash is a function of the SAME prefix order every archived model saw.
    assert FEATURE_V1_NAMES == tuple(
        n for n in DEFAULT_FEATURES if n not in VOLUME_FLOW_FEATURES)
    assert len(FEATURE_V1_NAMES) == 39
    assert set(FEATURE_NAMES) == set(FEATURE_V1_NAMES) | set(VOLUME_FLOW_FEATURES)
    assert not (set(FEATURE_V1_NAMES) & set(VOLUME_FLOW_FEATURES))


def test_the_declared_v1_hash_is_recomputed_from_the_v1_list():
    """Recompute it here — a hand-typed literal may not be trusted (D-4)."""
    from core.ml.features import (FEATURE_SCHEMA_V1_HASH, FEATURE_V1_NAMES,
                                  feature_schema_hash)
    recomputed = hashlib.sha1(
        json.dumps(list(FEATURE_V1_NAMES)).encode("utf-8")).hexdigest()[:12]
    assert recomputed == FEATURE_SCHEMA_V1_HASH
    assert feature_schema_hash(FEATURE_V1_NAMES) == FEATURE_SCHEMA_V1_HASH
    # ... and the value every model trained before P6-B carries.
    assert FEATURE_SCHEMA_V1_HASH == "335e63360104"
    assert feature_schema_hash() == "1f30fded996d"      # the live v2 contract


def test_the_research_documents_two_hashes_reproduce():
    """D-4's own arithmetic: 39 names → 335e63360104, 40 (+``hurst``) → 70899cff156d.

    The 40-name list is the v1 list with bare ``hurst`` re-inserted where the
    pre-P6 header had it (immediately before ``hurst_signal``), which is the list
    that revision's ``DEFAULT_FEATURES`` actually contained.
    """
    from core.ml.features import FEATURE_V1_NAMES

    def h(names):
        return hashlib.sha1(json.dumps(list(names)).encode("utf-8")
                            ).hexdigest()[:12]

    with_hurst = list(FEATURE_V1_NAMES)
    with_hurst.insert(with_hurst.index("hurst_signal"), "hurst")
    assert len(with_hurst) == 40
    assert h(FEATURE_V1_NAMES) == "335e63360104"
    assert h(with_hurst) == "70899cff156d"


def test_the_v1_refusal_path_still_names_v1():
    """The frozen hash is what makes a pre-P6 model refused *by name*."""
    from core.ml.features import (FEATURE_SCHEMA_V1_HASH,
                                  feature_schema_label,
                                  feature_schema_mismatch_reason)
    assert feature_schema_label(FEATURE_SCHEMA_V1_HASH).startswith("v1")
    reason = feature_schema_mismatch_reason(FEATURE_SCHEMA_V1_HASH)
    assert "schema hash mismatch" in reason
    assert "v1" in reason


@pytest.mark.parametrize("names", [("ret_1",), ("ret_1", "ret_5")])
def test_the_hashing_rule_is_one_rule(names):
    from core.ml.features import feature_schema_hash
    assert feature_schema_hash(list(names)) == hashlib.sha1(
        json.dumps(list(names)).encode("utf-8")).hexdigest()[:12]
