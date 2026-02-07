import os
import uuid
import logging
import duckdb
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

from advanced_catdap.service.schema import DatasetMetadata, ColumnInfo

class DatasetManager:
    def __init__(self, storage_dir: Union[str, Path] = "data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def _get_connection(self):
        return duckdb.connect(database=':memory:')

    def _build_metadata_from_parquet(
        self, target_path: Path, dataset_id: str, filename: str
    ) -> DatasetMetadata:
        con = self._get_connection()
        try:
            rel = con.from_parquet(str(target_path))
            n_rows = rel.count("*").fetchone()[0]
            dtypes = rel.types
            col_names = rel.columns

            cols_info = []
            for i, col in enumerate(col_names):
                stats = con.execute(
                    f"""
                    SELECT
                        COUNT("{str(col).replace('"', '""')}") as valid_count,
                        APPROX_COUNT_DISTINCT("{str(col).replace('"', '""')}") as approx_unique
                    FROM '{target_path}'
                """
                ).fetchone()
                valid_count, unique_approx = stats
                missing_count = n_rows - valid_count
                cols_info.append(
                    ColumnInfo(
                        name=col,
                        dtype=str(dtypes[i]),
                        missing_count=missing_count,
                        unique_approx=unique_approx,
                    )
                )

            return DatasetMetadata(
                dataset_id=dataset_id,
                filename=filename,
                file_path=str(target_path.absolute()),
                n_rows=n_rows,
                n_columns=len(col_names),
                columns=cols_info,
                created_at=datetime.now(),
            )
        finally:
            con.close()

    def register_dataset(self, file_path: Union[str, Path], dataset_id: Optional[str] = None, original_filename: Optional[str] = None) -> DatasetMetadata:
        """
        Register a dataset (CSV/Parquet) into the managed storage (Parquet).
        Returns metadata.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if dataset_id is None:
            dataset_id = str(uuid.uuid4())

        target_path = self.storage_dir / f"{dataset_id}.parquet"
        
        con = self._get_connection()
        try:
            # Detect format and load
            if file_path.suffix.lower() == '.csv':
                read_cmd = f"read_csv_auto('{file_path}')"
            elif file_path.suffix.lower() == '.parquet':
                read_cmd = f"read_parquet('{file_path}')"
            else:
                 raise ValueError("Unsupported format. Only CSV and Parquet are supported.")

            # Copy to storage if not already there (or if converting)
            con.execute(f"COPY (SELECT * FROM {read_cmd}) TO '{target_path}' (FORMAT PARQUET)")
            
            metadata = self._build_metadata_from_parquet(
                target_path=target_path,
                dataset_id=dataset_id,
                filename=original_filename or file_path.name,
            )
            
            self.logger.info(
                "Dataset registered: %s (%s rows)", dataset_id, metadata.n_rows
            )
            return metadata

        except Exception as e:
            self.logger.error(f"Failed to register dataset: {e}")
            if target_path.exists(): 
                try: os.remove(target_path) 
                except: pass
            raise e
        finally:
            con.close()

    def get_sample(self, dataset_id: str, n_rows: int = 100000, seed: int = 42, stratify_col: Optional[str] = None) -> pd.DataFrame:
        """
        Get a sample of the dataset as a Pandas DataFrame.
        """
        target_path = self.storage_dir / f"{dataset_id}.parquet"
        if not target_path.exists():
             raise FileNotFoundError(f"Dataset {dataset_id} not found.")

        con = self._get_connection()
        try:
            query = f"SELECT * FROM '{target_path}'"
            
            # DuckDB sampling - simplified
            df = con.execute(f"""
                SELECT * FROM '{target_path}' USING SAMPLE {n_rows} ROWS
            """).df()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Sampling failed: {e}")
            raise e
        finally:
            con.close()

    def get_preview(self, dataset_id: str, n_rows: int = 100) -> pd.DataFrame:
        target_path = self.storage_dir / f"{dataset_id}.parquet"
        if not target_path.exists():
             raise FileNotFoundError(f"Dataset {dataset_id} not found.")
        
        con = self._get_connection()
        try:
            return con.execute(f"SELECT * FROM '{target_path}' LIMIT {n_rows}").df()
        finally:
            con.close()

    def _get_row_count(self, path: Path) -> int:
        con = self._get_connection()
        try:
            return con.from_parquet(str(path)).count('*').fetchone()[0]
        finally:
            con.close()

    def get_dataset_metadata(self, dataset_id: str, filename: Optional[str] = None) -> DatasetMetadata:
        target_path = self.storage_dir / f"{dataset_id}.parquet"
        if not target_path.exists():
            raise FileNotFoundError(f"Dataset {dataset_id} not found.")
        return self._build_metadata_from_parquet(
            target_path=target_path,
            dataset_id=dataset_id,
            filename=filename or target_path.name,
        )
