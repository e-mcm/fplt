"""Pandera.polars and pydantic schemas for tables that cross module boundaries.

Data-model contracts: ``manifests`` (runs, builds, official including
``SubmittedRecord``, experiments), ``l2``, ``outcomes``, ``squad_state_table``,
``experiment_results``. Official points PMFs live in
``fplt.store.points_pmf``. Ops-06 quarantine rows:
``quarantine.QuarantineRecordSchema``. Ops-08 weekly report rows:
``weekly_report.WeeklyReportSchema``. Derived catalog:
``catalog.CatalogRunSchema`` (Stage C / step 8). Ops-10 health document:
``status.StatusDocument``. Timing JSON: ``timing.TimingRecord``.
"""
