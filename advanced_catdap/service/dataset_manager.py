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

    @staticmethod
    def _quote_identifier(name: str) -> str:
        """Quote SQL identifiers safely for DuckDB."""
        return '"' + str(name).replace('"', '""') + '"'

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
                rel = con.read_csv(str(file_path), header=True)
            elif file_path.suffix.lower() == '.parquet':
                rel = con.from_parquet(str(file_path))
            else:
                 raise ValueError("Unsupported format. Only CSV and Parquet are supported.")

            # Copy to storage if not already there (or if converting)
            rel.to_parquet(str(target_path))
            
            # Analyze metadata using DuckDB
            rel = con.from_parquet(str(target_path))
            n_rows = rel.count('*').fetchone()[0]
            
            # Get column types
            dtypes = rel.types
            col_names = rel.columns
            
            cols_info = []
            for i, col in enumerate(col_names):
                # Calculate simple stats
                quoted = self._quote_identifier(col)
                stats = rel.aggregate(
                    f"COUNT({quoted}) AS valid_count, APPROX_COUNT_DISTINCT({quoted}) AS approx_unique"
                ).fetchone()
                
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
                filename=original_filename or file_path.name,
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
