"""Pandera.polars and pydantic schemas for tables that cross module boundaries.

Data-model contracts: ``manifests`` (runs, builds, official, experiments),
``l2``, ``outcomes``, ``squad_state_table``, ``experiment_results``. Ops-06
quarantine rows: ``quarantine.QuarantineRecordSchema``. Derived catalog:
``catalog.CatalogRunSchema`` (Stage C / step 8).
"""
