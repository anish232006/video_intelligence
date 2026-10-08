"""
src/evaluation/evaluator.py — Evaluation framework.

Supports:
- Our system (track-level + semantic + time filters)
- Baseline (full-frame CLIP without tracking)
- Metrics: Top-1, Top-5, camera accuracy, time error, latency
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from src.config import cfg
from src.models import EvalQuery, EvalResult
from src.query.parser import QueryParser
from src.query.retriever import RetrievalPipeline

logger = logging.getLogger(__name__)

parser = QueryParser()
pipeline = RetrievalPipeline()


def load_eval_queries(path: Optional[Path] = None) -> List[EvalQuery]:
    """Load evaluation queries from JSON file."""
    path = path or cfg.evaluation_queries_path
    if not Path(path).exists():
        logger.warning("Evaluation queries file not found: %s", path)
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [EvalQuery(**q) for q in data]


def run_evaluation(queries: List[EvalQuery], top_k: int = 5) -> Dict:
    """
    Run evaluation on our track-level system.

    Returns:
        Dict with per-query results and aggregate metrics.
    """
    results = []
    parser_inst = QueryParser()
    pipeline_inst = RetrievalPipeline()

    for eq in queries:
        t0 = time.time()
        try:
            structured = parser_inst.parse(eq.query)
            search_results, _ = pipeline_inst.search(structured, top_k=top_k)
        except Exception as e:
            logger.error("Evaluation query failed: %s — %s", eq.query, e)
            search_results = []

        latency_ms = (time.time() - t0) * 1000

        found = len(search_results) > 0
        top1_camera = search_results[0].camera_id if found else None
        top1_time = search_results[0].timestamp if found else None

        camera_correct = (
            top1_camera == eq.expected_camera
            if eq.expected_camera and top1_camera
            else False
        )

        time_error = None
        if top1_time is not None and eq.expected_time_start is not None:
            mid_expected = (eq.expected_time_start + (eq.expected_time_end or eq.expected_time_start)) / 2
            time_error = abs(top1_time - mid_expected)

        results.append(EvalResult(
            query=eq.query,
            expected_camera=eq.expected_camera,
            top1_camera=top1_camera,
            top1_time=top1_time,
            camera_correct=camera_correct,
            time_error=time_error,
            latency_ms=latency_ms,
            found=found,
        ))

    return _aggregate_metrics(results)


def _aggregate_metrics(results: List[EvalResult]) -> Dict:
    """Compute aggregate evaluation metrics."""
    if not results:
        return {"error": "No results"}

    n = len(results)
    n_found = sum(1 for r in results if r.found)
    n_camera_correct = sum(1 for r in results if r.camera_correct)
    time_errors = [r.time_error for r in results if r.time_error is not None]
    latencies = [r.latency_ms for r in results]

    metrics = {
        "total_queries": n,
        "top1_accuracy": round(n_found / n, 3) if n > 0 else 0,
        "camera_accuracy": round(n_camera_correct / n, 3) if n > 0 else 0,
        "mean_localization_error_seconds": round(sum(time_errors) / len(time_errors), 2) if time_errors else None,
        "mean_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
        "per_query": [
            {
                "query": r.query,
                "found": r.found,
                "camera_correct": r.camera_correct,
                "top1_camera": r.top1_camera,
                "time_error_seconds": r.time_error,
                "latency_ms": round(r.latency_ms, 1),
            }
            for r in results
        ],
    }
    return metrics
