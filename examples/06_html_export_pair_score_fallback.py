"""Minimal example: HTML export works when interaction_importances uses pair_score only.

Also demonstrates stable Global Interaction Network serialization for browser playback
by emitting plain JSON arrays (not typed-array bdata) for heatmap `z`.
"""

from pathlib import Path

from advanced_catdap.service.exporter import ResultExporter


def main() -> None:
    result = {
        "mode": "CLASSIFICATION",
        "baseline_score": 123.0,
        "feature_importances": [
            {"Feature": "A", "Delta_Score": 5.0, "Score": 118.0},
            {"Feature": "B", "Delta_Score": 2.0, "Score": 121.0},
        ],
        "interaction_importances": [
            {"feature_1": "A", "feature_2": "B", "pair_score": 11.25}
        ],
        "feature_details": {},
        "interaction_details": {},
    }
    html_io = ResultExporter.generate_html_report(result, meta={"dataset_id": "pair_score_demo"})
    out_path = Path("analysis/reports/pair_score_fallback_demo.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(html_io.getvalue())
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
