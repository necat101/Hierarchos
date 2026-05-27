import json
import tempfile
import unittest
from pathlib import Path

from hierarchos.evaluation.arc_agi import extract_json_grid, load_arc_agi_tasks
from hierarchos.evaluation.benchmarks import get_benchmark, resolve_task_names


class BenchmarkRegistryTests(unittest.TestCase):
    def test_frontier_text_expands_to_runnable_tasks_and_arc_agi_external(self):
        tasks, external, unknown = resolve_task_names(["frontier-text"], include_external=True)

        self.assertIn("mmlu_pro", tasks)
        self.assertIn("gpqa_diamond_n_shot", tasks)
        self.assertIn("aime25", tasks)
        self.assertIn("bbh_cot_fewshot", tasks)
        self.assertIn("arc_agi", {spec.key for spec in external})
        self.assertNotIn("arc_agi", tasks)
        self.assertEqual(unknown, [])

    def test_arc_agi_alias_is_first_class_external_benchmark(self):
        spec = get_benchmark("arc-agi")

        self.assertIsNotNone(spec)
        self.assertEqual(spec.key, "arc_agi")
        self.assertFalse(spec.runnable)

        tasks, external, unknown = resolve_task_names(["arc-agi"], include_external=True)
        self.assertEqual(tasks, [])
        self.assertEqual([spec.key for spec in external], ["arc_agi"])
        self.assertEqual(unknown, [])

    def test_arc_agi_family_includes_official_interactive_route(self):
        tasks, external, unknown = resolve_task_names(["arc-agi-family"], include_external=True)

        self.assertEqual(tasks, [])
        self.assertEqual([spec.key for spec in external], ["arc_agi", "arc_agi_2", "arc_agi_3"])
        self.assertEqual(unknown, [])

    def test_raw_lm_eval_tasks_are_preserved_for_new_harness_plugins(self):
        tasks, external, unknown = resolve_task_names(["new_custom_task"], include_external=True)

        self.assertEqual(tasks, ["new_custom_task"])
        self.assertEqual(external, [])
        self.assertEqual(unknown, ["new_custom_task"])

    def test_arc_agi_json_helpers_parse_grid_and_task_file(self):
        self.assertEqual(extract_json_grid("answer: [[1,2],[3,4]]"), [[1, 2], [3, 4]])

        with tempfile.TemporaryDirectory() as tmp_dir:
            task_path = Path(tmp_dir) / "sample.json"
            task_path.write_text(
                json.dumps(
                    {
                        "train": [{"input": [[0]], "output": [[1]]}],
                        "test": [{"input": [[2]], "output": [[3]]}],
                    }
                ),
                encoding="utf-8",
            )

            tasks = load_arc_agi_tasks(str(task_path))

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].task_id, "sample")
        self.assertEqual(tasks[0].train[0].input, [[0]])
        self.assertEqual(tasks[0].test[0].output, [[3]])


if __name__ == "__main__":
    unittest.main()
