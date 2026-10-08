"""DuckDB-backed store.

How this scales, and why it is built this way:

A CSV can be registered as a VIEW -- then nothing is held in memory, but every query
re-parses the whole file. Profiling a 32 MB CSV that way took 24 seconds, because the
profile runs ~30 queries and each one re-read the file.

So CSV and JSON are INGESTED ONCE into a DuckDB table backed by a file on disk. Parsing
happens a single time; afterwards every query hits DuckDB's compressed columnar format.
RAM stays low because the database is on disk, not in memory.

Parquet is already columnar, so it stays a view -- no ingest needed.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import duckdb
import pandas as pd


class Store:
    def __init__(self, db_path: str | Path | None = None) -> None:
        # file-backed, not :memory:, so a large ingest spills to disk instead of RAM
        self.db_path = Path(db_path) if db_path else (
            Path(tempfile.gettempdir()) / f"datastudio_{uuid.uuid4().hex[:8]}.duckdb"
        )
        self.con = duckdb.connect(str(self.db_path))
        self.table: str | None = None
        self.source: str | None = None
        self.mode: str | None = None       # "ingested" or "view"

    # ---------- registering ------------------------------------------------

    def register_file(self, path: str | Path, table: str = "data") -> str:
        """Make `path` queryable as `table`.

        Parquet -> view (already columnar).
        CSV / JSON / Excel -> ingested once into a DuckDB table.
        """
        path = str(path)
        low = path.lower()
        self.table, self.source = table, path

        if low.endswith(".parquet"):
            self.con.execute(
                f'CREATE OR REPLACE VIEW "{table}" AS SELECT * FROM read_parquet(?)', [path]
            )
            self.mode = "view"
            return table

        if low.endswith((".xlsx", ".xls")):
            df = pd.read_excel(path)          # no streaming reader exists for Excel
            self.con.register("tmp_excel", df)
            self.con.execute(f'CREATE OR REPLACE TABLE "{table}" AS SELECT * FROM tmp_excel')
            self.con.unregister("tmp_excel")
            self.mode = "ingested"
            return table

        if low.endswith((".json", ".ndjson")):
            reader = "read_json_auto(?)"
        else:
            # sample_size=-1 scans the whole file for type inference -- slower once,
            # but avoids wrong column types, the usual cause of silently wrong results.
            reader = "read_csv_auto(?, header=true, sample_size=-1)"

        self.con.execute(f'CREATE OR REPLACE TABLE "{table}" AS SELECT * FROM {reader}', [path])
        self.mode = "ingested"
        return table

    # ---------- introspection ---------------------------------------------

    def columns(self) -> list[tuple[str, str]]:
        rows = self.con.execute(f'DESCRIBE "{self.table}"').fetchall()
        return [(r[0], r[1]) for r in rows]

    def schema_text(self) -> str:
        """Tiny regardless of file size -- this is all the LLM ever sees."""
        cols = ", ".join(f"{n} ({t})" for n, t in self.columns())
        return f'Table "{self.table}" has columns: {cols}'

    def row_count(self) -> int:
        return self.con.execute(f'SELECT COUNT(*) FROM "{self.table}"').fetchone()[0]

    # ---------- execution --------------------------------------------------

    def run(self, sql: str) -> pd.DataFrame:
        """Run validated SQL. Callers MUST pass it through guards first."""
        return self.con.execute(sql).fetchdf()

    def head(self, n: int = 50) -> pd.DataFrame:
        return self.run(f'SELECT * FROM "{self.table}" LIMIT {n}')

    def close(self) -> None:
        self.con.close()
