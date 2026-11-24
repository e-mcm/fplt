"""Pandera.polars and pydantic schemas for tables that cross module boundaries.

Data-model contracts (Stage B2): ``manifests`` (RunManifest, BuildManifest,
OfficialPointer, AppliedRecord, InterventionRecord), ``l2`` (L2StandardKeys,
SCHEMA_VERSION), ``outcomes``, ``squad_state_table``. Experiment/batch
manifests wait for Stage C.
"""
