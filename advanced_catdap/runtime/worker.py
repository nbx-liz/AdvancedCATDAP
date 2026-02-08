from __future__ import annotations

import argparse


def invoke_worker(job_id: str, dataset_id: str, params: str, db_path: str, data_dir: str) -> None:
    """Lazy-import worker entry so GUI mode has no worker-only import cost."""
    from advanced_catdap.service.local_worker import run_worker

    run_worker(
        job_id=job_id,
        dataset_id=dataset_id,
        params_json=params,
        data_dir=data_dir,
        db_path=db_path,
    )


def run_worker_mode(args: argparse.Namespace) -> int:
    """Execute one worker job and exit."""
    invoke_worker(
        job_id=args.job_id,
        dataset_id=args.dataset_id,
        params=args.params,
        db_path=args.db_path,
        data_dir=args.data_dir,
    )
    return 0

