from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union

from datetime import datetime

class ColumnInfo(BaseModel):
    name: str
    dtype: str
    missing_count: int = 0
    unique_approx: int = 0

class DatasetMetadata(BaseModel):
    dataset_id: str
    filename: str
    file_path: str
    n_rows: int
    n_columns: int
    columns: List[ColumnInfo] = []
    created_at: datetime = Field(default_factory=datetime.now)

class FeatureImportance(BaseModel):
    feature: str = Field(..., alias="Feature")
    score: float = Field(..., alias="Score")
    delta_score: float = Field(..., alias="Delta_Score")
    actual_bins: int = Field(..., alias="Actual_Bins")
    method: str = Field(..., alias="Method")

class InteractionImportance(BaseModel):
    feature_1: str = Field(..., alias="Feature_1")
    feature_2: str = Field(..., alias="Feature_2")
    pair_score: float = Field(..., alias="Pair_Score")
    gain: float = Field(..., alias="Gain")

class FeatureDetail(BaseModel):
    bin_edges: Optional[List[float]] = None
    bin_labels: Optional[List[str]] = None
    bin_counts: Optional[List[int]] = None
    bin_means: Optional[List[float]] = None
    woe: Optional[List[float]] = None
    iv: Optional[float] = None

class AnalysisParams(BaseModel):
    target_col: str
    candidates: Optional[List[str]] = None
    task_type: str = "auto"
    max_bins: int = 5
    min_samples_per_bin: int = 10
    top_k: int = 20
    delta_threshold: float = 0.0
    force_categoricals: Optional[List[str]] = None
    sample_size: Optional[int] = None
    use_aicc: bool = True
    random_state: int = 42


class InteractionDetail(BaseModel):
    feature_1: str
    feature_2: str
    bin_labels_1: List[str]
    bin_labels_2: List[str]
    counts: List[List[int]] # 2D matrix
    means: List[List[float]] # 2D matrix

class AnalysisResult(BaseModel):
    job_id: Optional[str] = None
    dataset_id: Optional[str] = None
    mode: str = "unknown"
    n_rows_used: int = 0
    sampled: bool = False
    baseline_score: float = 0.0
    feature_importances: List[FeatureImportance] = []
    interaction_importances: List[InteractionImportance] = []
    transform_rules: Dict[str, Any] = {}
    feature_details: Dict[str, FeatureDetail] = {}
    interaction_details: Dict[str, InteractionDetail] = {} # Key: "Feat1|Feat2"
    artifacts: Dict[str, str] = {}
