"""0.16.0 카메라 시점·앵글 (view)."""
import unittest

from promptmaker import PromptMaker, view
from promptmaker.element_check import check_item


class ViewDetectTests(unittest.TestCase):
    def test_korean(self):
        self.assertEqual(view.detect("뒷모습, 로우앵글")[0]["direction"], "rear")
        self.assertEqual(view.detect("뒷모습, 로우앵글")[0]["height"], "low")
        self.assertEqual(view.detect("정면 사선에서 찍은 사진")[0]["direction"], "front_3q")
        self.assertEqual(view.detect("뒤 사선 구도")[0]["direction"], "rear_3q")
        self.assertEqual(view.detect("뒤돌아보는 여자")[0]["direction"], "rear_3q")
        self.assertEqual(view.detect("옆모습 하이앵글 더치 앵글")[0],
                         {"direction": "side", "height": "high", "dutch": "dutch", "facing": None})
        self.assertEqual(view.detect("바로 위에서 내려다보는 탑뷰")[0]["height"], "overhead")
        self.assertIsNone(view.detect("하늘을 올려다보는 여자")[0]["height"])  # 시선이지 카메라 높이가 아님

    def test_english(self):
        self.assertEqual(view.detect("", "rear three-quarter view")[0]["direction"], "rear_3q")
        self.assertEqual(view.detect("", "photo from behind, low angle")[0]["height"], "low")
        self.assertEqual(view.detect("", "worm's-eye view of a woman")[0]["height"], "worm")
        # 상호작용 · 조명 · 국적은 시점이 아님
        self.assertIsNone(view.detect("", "a man hugging her from behind")[0]["direction"])
        self.assertIsNone(view.detect("", "soft light from above")[0]["height"])
        self.assertIsNone(view.detect("", "a Dutch woman")[0]["dutch"])

    def test_ko_to_english(self):
        self.assertIn("rear view", view.ko_to_english("뒷모습 사진"))
        self.assertIn("low-angle shot", view.ko_to_english("로우앵글로"))
        self.assertEqual(view.ko_to_english("뒤돌아보는 여자"), "뒤돌아보는 여자")  # 동작 말은 번역기에 맡김

    def test_route_camera_line(self):
        r = check_item("camera", "shot from a low angle, looking up at her", keys=["camera", "view"])
        self.assertEqual((r["status"], r["key"]), ("moved", "view"))
        self.assertTrue(r["text"].endswith("#low"))
        self.assertEqual(check_item("camera", "shot on an 85mm lens at f/1.4", keys=["camera", "view"])["status"], "ok")


class ViewGenerateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.maker = PromptMaker()

    def test_rear_view_hides_face(self):
        for seed in range(12):
            r = self.maker.generate("a woman walking on the beach, rear view, low angle", seed=seed)
            p = r.base_prompt.lower()
            self.assertIn("view", r.picks)
            self.assertNotRegex(p, r"\b(?:at|into) the camera\b", r.base_prompt)
            self.assertNotIn("toward the camera", p, r.base_prompt)
            self.assertNotRegex(p, r"\bwink", r.base_prompt)

    def test_option_and_camera_lines(self):
        for seed in range(8):
            r = self.maker.generate("a woman in a bedroom", seed=seed, view="side", view_height="high")
            self.assertEqual(view.view_in_text(r.picks["view"]).get("direction"), "side")
            self.assertEqual(view.view_in_text(r.picks["view"]).get("height"), "high")
            cam = r.picks.get("camera", "")
            self.assertFalse(view.states_view(cam), cam)  # 각도를 말하는 카메라 줄은 빠짐

    def test_none_option(self):
        r = self.maker.generate("뒷모습 사진", seed=1, view="none", view_height="none")
        self.assertNotIn("view", r.picks)

    def test_three_quarter_describes_torso(self):
        r = self.maker.generate("a woman in a cafe, three-quarter view", seed=3)
        self.assertIn("three-quarter", r.picks["view"])
        self.assertIn("torso", r.picks["view"])  # 몸통이 어떻게 보이는지 함께 (사선이 측면으로 나오는 것 방지)

    def test_random_view_avoids_extremes(self):
        seen = set()
        for seed in range(60):
            r = self.maker.generate("", seed=seed, mode="random")
            v = r.lexicon["trace"]
            for line in v:
                if line.startswith("view:") and "[random]" in line:
                    seen.add(line)
                    self.assertNotRegex(line.lower(), r"rear|overhead|worm|dutch")
        self.assertTrue(seen)


if __name__ == "__main__":
    unittest.main()
