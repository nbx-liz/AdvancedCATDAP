import httpx
from typing import Optional, Dict, Any, List
from advanced_catdap.service.schema import DatasetMetadata, AnalysisParams, AnalysisResult

class APIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def upload_dataset(self, file_obj, filename: str) -> DatasetMetadata:
        files = {"file": (filename, file_obj, "application/octet-stream")}
        try:
            resp = httpx.post(f"{self.base_url}/datasets", files=files, timeout=30.0)
            resp.raise_for_status()
            return DatasetMetadata(**resp.json())
        except httpx.HTTPError as e:
            raise RuntimeError(f"Upload failed: {e}")

    def get_dataset(self, dataset_id: str) -> DatasetMetadata:
        try:
            resp = httpx.get(f"{self.base_url}/datasets/{dataset_id}")
            resp.raise_for_status()
            return DatasetMetadata(**resp.json())
        except httpx.HTTPError as e:
            raise RuntimeError(f"Fetch metadata failed: {e}")

    def get_preview(self, dataset_id: str, rows: int = 100) -> List[Dict[str, Any]]:
        try:
            resp = httpx.get(f"{self.base_url}/datasets/{dataset_id}/preview", params={"rows": rows})
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            raise RuntimeError(f"Fetch preview failed: {e}")

    def submit_job(self, dataset_id: str, params: AnalysisParams) -> str:
        try:
            # Pydantic -> JSON
            payload = params.model_dump()
            resp = httpx.post(
                f"{self.base_url}/jobs", 
                params={"dataset_id": dataset_id}, 
                json=payload
            )
            resp.raise_for_status()
            return resp.json()["job_id"]
        except httpx.HTTPError as e:
            raise RuntimeError(f"Job submission failed: {e}")

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        try:
            resp = httpx.get(f"{self.base_url}/jobs/{job_id}")
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            raise RuntimeError(f"Job status check failed: {e}")

    def export_html_report(
        self,
        result: Dict[str, Any],
        meta: Optional[Dict[str, Any]],
        filename: str,
        theme: str = "dark",
    ) -> Dict[str, Any]:
        payload = {
            "result": result,
            "meta": meta or {},
            "filename": filename,
            "theme": theme,
        }
        try:
            resp = httpx.post(f"{self.base_url}/export/html", json=payload, timeout=60.0)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            raise RuntimeError(f"HTML export failed: {e}")
