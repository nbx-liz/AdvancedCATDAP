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
        # Lightweight connection for metadata operations
        self.con = duckdb.connect(database=':memory:')

    def register_dataset(self, file_path: Union[str, Path], dataset_id: Optional[str] = None) -> DatasetMetadata:
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
        
        # Use DuckDB to convert/copy to Parquet efficiently
        try:
            # Detect format and load
            if file_path.suffix.lower() == '.csv':
                read_cmd = f"read_csv_auto('{file_path}')"
            elif file_path.suffix.lower() == '.parquet':
                read_cmd = f"read_parquet('{file_path}')"
            else:
                 raise ValueError("Unsupported format. Only CSV and Parquet are supported.")

            # Copy to storage if not already there (or if converting)
            self.con.execute(f"COPY (SELECT * FROM {read_cmd}) TO '{target_path}' (FORMAT PARQUET)")
            
            # Analyze metadata using DuckDB
            # 1. Basic stats (count, columns)
            # 2. Detailed stats (approx_distinct, valid_count) for schema
            
            # Relies on the newly created parquet file
            rel = self.con.from_parquet(str(target_path))
            n_rows = rel.count('*').fetchone()[0]
            
            # Get column types
            # duckdb types need mapping to simple string types
            dtypes = rel.types
            col_names = rel.columns
            
            cols_info = []
            for i, col in enumerate(col_names):
                # Calculate simple stats
                # Using SQL for efficiency on large files
                stats = self.con.execute(f"""
                    SELECT 
                        COUNT({col}) as valid_count, 
                        APPROX_COUNT_DISTINCT({col}) as approx_unique 
                    FROM '{target_path}'
                """).fetchone()
                
                valid_count, unique_approx = stats
                missing_count = n_rows - valid_count
                
                cols_info.append(ColumnInfo(
                    name=col,
                    dtype=str(dtypes[i]),
                    missing_count=missing_count,
                    unique_approx=unique_approx
                ))

            metadata = DatasetMetadata(
                dataset_id=dataset_id,
                filename=file_path.name,
                file_path=str(target_path.absolute()),
                n_rows=n_rows,
                n_columns=len(col_names),
                columns=cols_info,
                created_at=datetime.now()
            )
            
            self.logger.info(f"Dataset registered: {dataset_id} ({n_rows} rows)")
            return metadata

        except Exception as e:
            self.logger.error(f"Failed to register dataset: {e}")
            if target_path.exists(): 
                try: os.remove(target_path) 
                except: pass
            raise e

    def get_sample(self, dataset_id: str, n_rows: int = 100000, seed: int = 42, stratify_col: Optional[str] = None) -> pd.DataFrame:
        """
        Get a sample of the dataset as a Pandas DataFrame.
        """
        target_path = self.storage_dir / f"{dataset_id}.parquet"
        if not target_path.exists():
             raise FileNotFoundError(f"Dataset {dataset_id} not found.")

        try:
            query = f"SELECT * FROM '{target_path}'"
            
            # DuckDB sampling
            # BERNOULLI sampling is fast and standard
            percent = (n_rows * 100.0) / self._get_row_count(target_path)
            # Cap at 100%
            percent = min(percent, 100.0)
            
            if percent >= 100.0:
                 return self.con.from_parquet(str(target_path)).df()
            
            # Note: DuckDB TABLESAMPLE is approximate. 
            # For exact stratified sampling, we might need more complex queries.
            # Impl: Simple random sample for now unless stratify is needed
            
            if stratify_col:
                 # Advanced stratify logic is expensive in SQL without window functions
                 # Simplified: Just grab random sample provided n_rows is sufficient
                 pass

            # deterministic sampling with seed in duckdb is tricky across versions
            # trying `USING SAMPLE reservoir(n ROWS)` if supported or `ORDER BY hash(uuid())`
            
            df = self.con.execute(f"""
                SELECT * FROM '{target_path}' USING SAMPLE {n_rows} ROWS
            """).df()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Sampling failed: {e}")
            raise e

    def get_preview(self, dataset_id: str, n_rows: int = 100) -> pd.DataFrame:
        target_path = self.storage_dir / f"{dataset_id}.parquet"
        if not target_path.exists():
             raise FileNotFoundError(f"Dataset {dataset_id} not found.")
        
        return self.con.execute(f"SELECT * FROM '{target_path}' LIMIT {n_rows}").df()

    def _get_row_count(self, path: Path) -> int:
        return self.con.from_parquet(str(path)).count('*').fetchone()[0]
