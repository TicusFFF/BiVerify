"""Exercise the same inference-to-scoring entry on demo and benchmark inputs."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ENTRY = next((ROOT / "0-3_all_model_test").rglob("run_biverify.py"))
sys.path.insert(0, str(ROOT))
import biverify_runner


class ReleaseEntryTest(unittest.TestCase):
    def test_shared_dataset_entry(self):
        spec = importlib.util.spec_from_file_location("flash_entry", ENTRY)
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        for dataset in ("demo", "hmmt_feb_2025"):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as work:
                base = Path(work)
                out = base / "run"
                data_root = entry.DATA_ROOT if dataset == "demo" else base / "datasets"
                questions = data_root / entry.DATASETS[dataset]["file"]
                if dataset != "demo":
                    questions.parent.mkdir(parents=True)
                    questions.write_text(json.dumps([{"q": "What is six times seven?", "answer": "42"}]), encoding="utf-8")
                row = json.loads(questions.read_text(encoding="utf-8"))[0]
                query = row["q"]
                sample_id = f"{dataset}_0"
                fixture = {"id": sample_id, "query": query, "response": "fixture", "pred": "",
                           "finish_reason": "length"}
                (base / f"deepseek-flash_{dataset}_first_response.jsonl").write_text(json.dumps(fixture)+"\n", encoding="utf-8")
                def digest(text):
                    return hashlib.sha256(text.encode("utf-8")).hexdigest()

                provenance = base / f"deepseek-flash_{dataset}_first_response.version.json"
                call = {"requested_model": "deepseek-flash", "returned_model": "deepseek-flash",
                        "prompt_sha256": digest(query), "response_sha256": digest(fixture["response"])}
                state = {"contract": {"model": "deepseek-flash", "temperature": 0,
                                      "dataset": dataset, "endpoint": "https://api.deepseek.com",
                                      "thinking": "disabled"},
                         "records": {sample_id: {"row": fixture, "calls": [call], "prediction_sha256": digest("")}}}
                provenance.write_text(json.dumps(state), encoding="utf-8")

                def verifier_fixture(*, output_path, dataset_key, **kwargs):
                    self.assertEqual(dataset_key, dataset)
                    output_path.write_text(json.dumps({"logs": [
                        {"id": sample_id, "query": query, "final_response": f"Final answer: {row['answer']}"}
                    ]}), encoding="utf-8")

                argv = [str(ENTRY), "--dataset", dataset, "--limit", "1", "--initial-dir", str(base),
                        "--output", str(out), "--run"]
                env = {"DEEPSEEK_API_KEY": "offline", "HTTPS_PROXY": "http://127.0.0.1:1", "NO_PROXY": ""}
                with patch.dict(os.environ, env), patch.object(sys, "argv", argv), patch.object(entry, "DATA_ROOT", data_root):
                    with patch.object(biverify_runner, "run_dataset", side_effect=verifier_fixture) as run:
                        entry.main()
                        run.assert_called_once()
                scores = json.loads((out / "evaluation.json").read_text(encoding="utf-8"))
                self.assertEqual([r["id"] for r in scores["rows"]], [sample_id])
                self.assertEqual((scores["acc"], scores["n_completed"]), (100.0, 1))
                state["contract"]["model"] = "different-model"
                provenance.write_text(json.dumps(state), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "expected initial answers from official DeepSeek Flash"):
                    entry.checked_inputs(dataset, questions,
                                         base / f"deepseek-flash_{dataset}_first_response.jsonl", 1)


if __name__ == "__main__":
    unittest.main()
