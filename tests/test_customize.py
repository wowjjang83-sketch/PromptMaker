"""[0.18.0] 프롬프트 커스터마이징 (customize.schema / build)."""
import re
import unittest

from promptmaker import PromptMaker
from promptmaker.customize import CATEGORIES, build, build_batch, schema


class CustomizeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pm = PromptMaker()

    def test_schema_has_steps_and_items(self):
        info = schema(self.pm)
        self.assertEqual([s["id"] for s in info["steps"]], ["subject", "place", "light", "camera"])
        cats = info["categories"]
        self.assertTrue(cats["hair"]["items"]["female"])
        self.assertTrue(cats["hair"]["items"]["male"])
        self.assertTrue(cats["outfit"]["items"]["female"])
        self.assertTrue(cats["hair_length"]["items"]["female"])
        for it in cats["nationality"]["items"]["female"]:
            self.assertIn(it["text"], ("Korean", "Japanese"))

    def test_auto_build_follows_rules(self):
        for seed in range(30):
            p = build(self.pm, {}, seed=seed)["prompt"]
            m = re.search(r"(\d+)-year-old (\w+) woman", p)
            self.assertIsNotNone(m, p)
            self.assertTrue(20 <= int(m.group(1)) <= 22, p)
            self.assertIn(m.group(2), ("Korean", "Japanese"))
            self.assertNotIn("{", p)

    def test_pick_and_off(self):
        spec = {"subjects": [{"gender": "female", "sel": {
            "hair": {"mode": "pick", "values": ["long wavy chestnut hair"]},
            "outfit": {"mode": "pick", "values": ["a satin camisole", "a pleated tennis skirt"]},
            "action": {"mode": "off"}, "expression": {"mode": "off"}, "pose": {"mode": "off"},
            "age": {"mode": "pick", "values": ["30"]}}}],
            "scene": {"location": {"mode": "pick", "values": ["in a cozy cafe with wooden interiors"]}}}
        p = build(self.pm, spec, seed=1)["prompt"]
        self.assertIn("long wavy chestnut hair", p)
        self.assertIn("a satin camisole and a pleated tennis skirt", p)
        self.assertIn("in a cozy cafe", p)
        self.assertIn("22-year-old", p)  # 여성 나이 규칙으로 보정
        self.assertNotIn(" is ", p.split("The scene")[0].split("wears")[-1])

    def test_two_subjects(self):
        spec = {"subjects": [{"gender": "female"}, {"gender": "male"}], "scene": {"view": {"direction": "rear"}}}
        p = build(self.pm, spec, seed=3)["prompt"]
        self.assertIn("two people", p)
        self.assertIn("The man", p)
        self.assertIn("rear view", p)

    def test_batch_differs(self):
        res = build_batch(self.pm, {}, count=3, seed=10)
        self.assertEqual(len(res), 3)
        self.assertGreater(len({r["prompt"] for r in res}), 1)

    def test_every_category_known(self):
        info = schema(self.pm)
        for step in info["steps"]:
            for g in step["groups"]:
                for c in g["categories"]:
                    self.assertIn(c, CATEGORIES)


if __name__ == "__main__":
    unittest.main()
