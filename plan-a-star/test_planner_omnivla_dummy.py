"""Dummy OmniVLA + tiny graph; mock Gemini. Run: python test_planner_omnivla_dummy.py"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import sys

_PEXP = Path(__file__).resolve().parents[1]
if str(_PEXP) not in sys.path:
    sys.path.insert(0, str(_PEXP))

from graph import Graph, Node
from graph_plan import Planner
from omnivla.uncertainty_heuristic import OmniVLAUncertaintyEstimator


def _write_jpg(path: Path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color).save(path, format="JPEG")


def _minimal_graph(tmp: Path) -> Graph:
    for i, c in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        _write_jpg(tmp / f"n{i}.jpg", c)
    data = {
        "scenes": ["dummy"],
        "sparsification_steps": 1,
        "drop_modality_p": 0.0,
        "nodes": [
            {
                "node_id": 0,
                "info": {"id": 0, "image": str(tmp / "n0.jpg")},
                "modalities": ["V"],
                "subnodes": {"V": {"image": str(tmp / "n0.jpg")}},
                "connections": [
                    {"modality": "V", "other_node_id": 1, "other_modality": "V"},
                ],
            },
            {
                "node_id": 1,
                "info": {"id": 1, "image": str(tmp / "n1.jpg")},
                "modalities": ["V"],
                "subnodes": {"V": {"image": str(tmp / "n1.jpg")}},
                "connections": [
                    {"modality": "V", "other_node_id": 2, "other_modality": "V"},
                ],
            },
            {
                "node_id": 2,
                "info": {"id": 2, "image": str(tmp / "n2.jpg")},
                "modalities": ["V"],
                "subnodes": {"V": {"image": str(tmp / "n2.jpg")}},
                "connections": [],
            },
        ],
    }
    p = tmp / "g.json"
    p.write_text(json.dumps(data))
    return Graph.deserialize(str(p))


def main():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        g = _minimal_graph(tmp)
        est = OmniVLAUncertaintyEstimator(None, use_dummy=True, dummy_mse=0.1)
        planner = Planner(
            g,
            omnivla_estimator=est,
            gamma_omnivla=0.5,
            planner_modality=(True, True),
        )
        with patch.object(Planner, "combined_grade", return_value=(100, 0)):
            plan = planner.plan(g.nodes[0], "V", g.nodes[2], "go to goal")
        assert plan is not None, "expected a path with high heuristic"
        print("ok:", plan)


if __name__ == "__main__":
    main()
