"""피사체 거리(0.14.0) 와 비율 추천(suggest_aspect) 테스트."""
import unittest

from promptmaker import NoTranslator, PromptMaker, suggest_aspect
from promptmaker import distance
from promptmaker.element_check import check_item


class DistanceDetectTests(unittest.TestCase):
    def test_korean_terms(self):
        self.assertEqual(distance.detect("멀리서 걷는 여자")[0], "far")
        self.assertEqual(distance.detect("원경 속 작은 여자")[0], "very_far")
        self.assertEqual(distance.detect("가까이서 찍은 사진")[0], "close")
        self.assertEqual(distance.detect("중거리에서")[0], "medium")
        self.assertIsNone(distance.detect("가까이 다가가는 여자")[0])  # 동작이지 거리가 아님

    def test_english_terms(self):
        self.assertEqual(distance.detect("", "photographed from a distance")[0], "far")
        self.assertEqual(distance.detect("", "up close portrait")[0], "close")
        self.assertEqual(distance.detect("", "an extreme long shot, tiny figure")[0], "very_far")
        self.assertIsNone(distance.detect("", "gazing out into the distance")[0])

    def test_framing_fit(self):
        self.assertFalse(distance.compatible("far", "extreme_closeup"))
        self.assertEqual(distance.fit_to_framing("far", "extreme_closeup"), "close")
        self.assertEqual(distance.fit_to_framing("far", "full_body"), "far")

    def test_route_phrase(self):
        self.assertEqual(distance.route_phrase("shot from a distance"), ("distance", "shot from a distance #far"))
        self.assertIsNone(distance.route_phrase("full-body shot from a distance"))  # 구도 문구는 framing
        r = check_item("camera", "shot from a distance", keys=["camera", "framing", "distance"])
        self.assertEqual((r["status"], r["key"]), ("moved", "distance"))


class DistanceGenerateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.maker = PromptMaker(translator=NoTranslator())

    def test_far_sets_full_body(self):
        r = self.maker.generate("a woman walking on the beach, shot from a distance", seed=1)
        self.assertIn("distance", r.picks)
        self.assertEqual(r.lexicon["intent"]["distance"], "far")
        self.assertIn("full", r.picks.get("framing", "").lower())

    def test_conflict_follows_framing(self):
        r = self.maker.generate("extreme close-up of a woman, shot from a distance", seed=2)
        self.assertTrue(any("피사체 거리" in w for w in r.warnings))
        self.assertIn(distance.distance_in_text(r.picks["distance"]) or "close", ("close",))

    def test_option_and_none(self):
        r = self.maker.generate("a woman in a cafe", seed=3, distance="very_far")
        self.assertIn("distance", r.picks)
        r = self.maker.generate("a woman in a cafe, shot from a distance", seed=3, distance="none")
        self.assertNotIn("distance", r.picks)
        with self.assertRaises(ValueError):
            self.maker.generate("a woman", distance="moon")

    def test_info_lists_distances(self):
        ids = [d["id"] for d in self.maker.info()["distances"]]
        self.assertEqual(ids[1:5], list(distance.DISTANCES))


class AspectTests(unittest.TestCase):
    def test_lying_is_landscape(self):
        self.assertEqual(suggest_aspect("She is lying on her side on a bed.")["ratio"], "16:9")

    def test_standing_full_body_is_portrait(self):
        r = suggest_aspect("Full-body shot showing her from head to toe. She is standing on a street.")
        self.assertEqual(r["orientation"], "portrait")

    def test_face_closeup_is_square(self):
        self.assertEqual(suggest_aspect("Extreme close-up of her face only. She smiles.")["ratio"], "1:1")

    def test_very_far_is_wide(self):
        r = suggest_aspect("Extreme long shot, her tiny figure within the expansive landscape. She is standing.")
        self.assertEqual(r["ratio"], "2:1")

    def test_empty_defaults_portrait(self):
        self.assertEqual(suggest_aspect("")["ratio"], "9:16")


if __name__ == "__main__":
    unittest.main()
