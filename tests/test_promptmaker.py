import json
import os
import tempfile
import threading
import unittest
import urllib.request

from promptmaker import GenerateOptions, NoTranslator, PromptMaker, SafetyError
from promptmaker import grammar, safety
from promptmaker.translator import DictionaryTranslator, is_english


class GrammarTests(unittest.TestCase):
    def test_articles(self):
        self.assertEqual(grammar.with_article("elegant woman"), "an elegant woman")
        self.assertEqual(grammar.with_article("unique look"), "a unique look")
        self.assertEqual(grammar.with_article("hour-long shoot"), "an hour-long shoot")
        self.assertEqual(grammar.fix_articles("a 85mm lens and an photo"), "an 85mm lens and a photo")

    def test_join_and_tidy(self):
        self.assertEqual(grammar.join_list(["a", "b", "c"]), "a, b, and c")
        self.assertEqual(grammar.tidy("hello , world .. next"), "Hello, world. Next")


class SafetyTests(unittest.TestCase):
    def test_minor_terms_rejected(self):
        for text in ["a teen girl", "17-year-old woman", "여고생", "schoolgirl outfit", "18살 여자"]:
            with self.assertRaises(SafetyError, msg=text):
                safety.check_text(text)

    def test_adult_terms_allowed(self):
        safety.check_text("a 25-year-old woman in a cafe")
        safety.check_text("스물다섯 살 여자")

    def test_age_clamp(self):
        self.assertEqual(safety.clamp_age("15"), 20)
        self.assertEqual(safety.clamp_age("27"), 27)


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pm = PromptMaker(translator=NoTranslator())

    def test_has_templates_and_elements(self):
        self.assertGreaterEqual(len(self.pm.templates), 5)
        self.assertLessEqual(len(self.pm.templates), 12)
        self.assertTrue(any(t.type == "video" for t in self.pm.templates.values()))
        self.assertGreater(len(self.pm.elements["location"].items), 20)

    def test_every_template_renders_cleanly(self):
        for tid in self.pm.templates:
            for seed in range(30):
                r = self.pm.generate("", mode="random_by_element", template=tid,
                                     random_elements=["*"], seed=seed)
                self.assertNotIn("{", r.prompt, (tid, r.prompt))
                self.assertNotIn("}", r.prompt, (tid, r.prompt))
                self.assertTrue(r.base_prompt[0].isupper())
                self.assertTrue(r.base_prompt.endswith("."))
                self.assertGreaterEqual(int(r.picks["age"]), 20)

    def test_seed_is_reproducible(self):
        a = self.pm.generate("beach", seed=42)
        b = self.pm.generate("beach", seed=42)
        self.assertEqual(a.prompt, b.prompt)

    def test_manual_mode_uses_selection(self):
        r = self.pm.generate("", mode="manual", template="portrait_default",
                             selections={"nationality": "Japanese", "location": "in a bamboo forest"})
        self.assertIn("Japanese", r.prompt)
        self.assertIn("in a bamboo forest", r.prompt)

    def test_random_by_type_respects_type(self):
        for seed in range(10):
            r = self.pm.generate("", mode="random_by_type", prompt_type="video", seed=seed)
            self.assertEqual(r.prompt_type, "video")

    def test_casting_filter(self):
        for seed in range(20):
            r = self.pm.generate("", mode="random_by_element", template="couple_romance",
                                 random_elements=["casting"], seed=seed)
            self.assertIn("man", r.picks["casting"])
            r = self.pm.generate("", mode="random_by_element", template="fashion_editorial",
                                 random_elements=["casting"], seed=seed)
            self.assertNotIn("man", r.picks["casting"].replace("woman", ""))

    def test_plural_grammar(self):
        r = self.pm.generate("", mode="manual", template="portrait_default",
                             selections={"casting": "two women"})
        self.assertIn("two", r.prompt)
        self.assertIn("They have", r.prompt)

    def test_keyword_boost(self):
        hits = sum("beach" in self.pm.generate("beach", seed=s, mode="random_by_element",
                                               template="portrait_default",
                                               random_elements=["location"]).picks["location"]
                   for s in range(20))
        self.assertGreater(hits, 8)

    def test_age_selection_clamped(self):
        r = self.pm.generate("", mode="manual", template="portrait_default", selections={"age": "16"})
        self.assertIn("20-year-old", r.prompt)
        self.assertTrue(r.warnings)

    def test_minor_input_rejected(self):
        with self.assertRaises(SafetyError):
            self.pm.generate("a cute teenager at school")

    def test_single_rating_all_items_available(self):
        # 수위 구분 없음: 예전 suggestive 항목도 같은 요소 파일에 있고, 옛 옵션은 무시된다
        texts = {i.text for i in self.pm.elements["attire_state"].items}
        self.assertIn("with a deep plunging neckline", texts)
        self.assertIn("a lace bodysuit", {i.text for i in self.pm.elements["lingerie"].items})
        self.assertNotIn("ratings", self.pm.info())
        r = self.pm.generate("", seed=1, content_rating="explicit")  # 알 수 없는 키는 무시
        self.assertTrue(r.prompt)

    def test_legacy_rating_files_merged(self):
        from promptmaker.data_loader import merge_legacy_rating_files
        with tempfile.TemporaryDirectory() as d:
            edir = os.path.join(d, "elements")
            os.makedirs(edir)
            with open(os.path.join(edir, "pose.txt"), "w", encoding="utf-8") as f:
                f.write("# 포즈\nleaning on a railing")  # 끝 줄바꿈 없음
            with open(os.path.join(edir, "pose.suggestive.txt"), "w", encoding="utf-8") as f:
                f.write("# 예전 수위 파일\nleaning on a railing\nlying on a silk sheet\n")
            with open(os.path.join(edir, "mood.explicit.txt"), "w", encoding="utf-8") as f:
                f.write("a quiet late-night mood\n")
            PromptMaker(user_data_dir=d, translator=NoTranslator())  # 만들 때 자동으로 합침
            self.assertEqual(sorted(os.listdir(edir)), ["mood.txt", "pose.txt"])
            with open(os.path.join(edir, "pose.txt"), encoding="utf-8") as f:
                pose = f.read()
            self.assertEqual(pose.count("leaning on a railing"), 1)
            self.assertIn("leaning on a railing\n# 수위 통합: pose.suggestive.txt 에서 옮김\nlying on a silk sheet\n", pose)
            self.assertEqual(merge_legacy_rating_files(d)["added"], 0)  # 다시 실행해도 안전

    def test_user_data_dir_extends(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "elements"))
            with open(os.path.join(d, "elements", "location.txt"), "w", encoding="utf-8") as f:
                f.write("# test\n10000::on the surface of the moon\n")
            pm = PromptMaker(user_data_dir=d, translator=NoTranslator())
            r = pm.generate("", mode="random_by_element", template="portrait_default",
                            random_elements=["location"], seed=1)
            self.assertIn("moon", r.prompt)


class IntentTests(unittest.TestCase):
    """The request must drive the result; random only fills unspecified slots."""

    @classmethod
    def setUpClass(cls):
        cls.pm = PromptMaker(translator=NoTranslator())

    def test_request_locks_elements(self):
        text = "A gravure photo taken by a 20-year-old Japanese model on the beach"
        for seed in range(25):
            for mode in ("random", "random_by_type"):
                r = self.pm.generate(text, seed=seed, mode=mode)
                self.assertEqual(r.prompt_type, "image")
                self.assertEqual(r.picks["age"], "20")
                self.assertEqual(r.picks["nationality"], "Japanese")
                self.assertEqual(r.picks["casting"], "one woman")
                self.assertIn("beach", r.picks.get("location", "beach"))
                self.assertIn("20-year-old Japanese model", r.prompt)

    def test_casting_from_request(self):
        for seed in range(10):
            r = self.pm.generate("two Korean women in a cafe", seed=seed)
            self.assertEqual(r.picks["casting"], "two women")
            self.assertIn("cafe", r.picks["location"])
            r = self.pm.generate("a couple at night", seed=seed)
            self.assertEqual(r.picks["casting"], "a woman and a man")

    def test_korean_age_and_video_type(self):
        pm = PromptMaker(translator=DictionaryTranslator(self.pm.ko_en), restrictions={"enabled": False})
        r = pm.generate("25살 한국 여자 영상", seed=1, mode="random")
        self.assertEqual(r.picks["age"], "25")
        # 기본 규칙(여성 20~22세)에서는 22 로 조정
        r = PromptMaker(translator=DictionaryTranslator(self.pm.ko_en)).generate("25살 한국 여자 영상", seed=1, mode="random")
        self.assertEqual(r.picks["age"], "22")
        self.assertEqual(r.prompt_type, "video")

    def test_user_selection_beats_request(self):
        pm = PromptMaker(translator=NoTranslator(), restrictions={"nationalities": ["Japanese", "Thai"]})
        r = pm.generate("a Japanese woman", seed=1, selections={"nationality": "Thai"})
        self.assertEqual(r.picks["nationality"], "Thai")

    def test_request_text_not_dumped(self):
        r = self.pm.generate("A gravure photo taken by a 20-year-old Japanese model on the beach", seed=3)
        self.assertNotIn("The scene captures", r.prompt)
        self.assertNotIn("year-old and", r.prompt)


class TranslatorTests(unittest.TestCase):
    def test_is_english(self):
        self.assertTrue(is_english("a woman in a cafe"))
        self.assertFalse(is_english("카페에 있는 여자"))

    def test_dictionary_fallback(self):
        pm = PromptMaker(translator=NoTranslator())
        t = DictionaryTranslator(pm.ko_en)
        out = t.translate("비 오는 날 카페")
        self.assertIn("cafe", out)
        self.assertIn("rain", out)

    def test_korean_input_via_dictionary(self):
        pm = PromptMaker(translator=None)
        pm.translator = DictionaryTranslator(pm.ko_en)
        r = pm.generate("해변 노을", seed=3)
        self.assertTrue(r.translated)
        self.assertIn("beach", r.lexicon["keywords"])


class ServerTests(unittest.TestCase):
    def test_api(self):
        from http.server import ThreadingHTTPServer
        from promptmaker.server import make_handler

        srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(PromptMaker(translator=NoTranslator())))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        try:
            info = json.loads(urllib.request.urlopen(base + "/api/info").read())
            self.assertIn("templates", info)
            req = urllib.request.Request(base + "/api/generate", method="POST",
                                         data=json.dumps({"user_input": "cafe", "count": 2}).encode())
            data = json.loads(urllib.request.urlopen(req).read())
            self.assertEqual(len(data["results"]), 2)
            html = urllib.request.urlopen(base + "/").read().decode()
            self.assertIn("PromptMaker", html)
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()


class CoherenceTests(unittest.TestCase):
    """Relation engine: the scene decides what fits (beach -> swimwear, never a bed)."""

    @classmethod
    def setUpClass(cls):
        cls.pm = PromptMaker(translator=NoTranslator())

    def _run(self, text, n=150, **kw):
        return [self.pm.generate(text, GenerateOptions(mode="random", seed=i, **kw)) for i in range(n)]

    def test_beach_gravure_never_bedroom_things(self):
        banned = ["bed ", "sheets", "pajama", "sleep shirt", "nightgown", "lingerie", "candle",
                  "sofa", "snow", "ankle boots", "stiletto", "tucked in", "dress flaring"]
        swim = 0
        for r in self._run("gravure photoshoot on the beach"):
            p = r.prompt.lower()
            for w in banned:
                self.assertNotIn(w, p, r.prompt)
            swim += any(w in p for w in ("bikini", "swimsuit", "monokini", "swim"))
        self.assertGreater(swim, 120)  # swimwear is the overwhelming choice

    def test_bedroom_has_no_beach_props(self):
        for r in self._run("photo in the bedroom", n=100):
            p = r.prompt.lower()
            for w in ("sand", "shoreline", "wading", "beach towel", "splashing water"):
                # 단어 단위로 확인 ('strappy sandals' 의 'sand' 는 모래가 아님)
                self.assertNotRegex(p, r"\b" + w + r"\b", r.prompt)

    def test_inline_tags_are_parsed(self):
        from promptmaker.data_loader import _parse_txt_line
        it = _parse_txt_line("2::lying on a beach towel #summer +beach !night")
        self.assertEqual(it.text, "lying on a beach towel")
        self.assertEqual((it.weight, it.tags, it.requires, it.excludes), (2.0, {"summer"}, {"beach"}, {"night"}))

    def test_conflicts_give_zero_weight(self):
        rel = self.pm.relations
        from promptmaker.data_loader import Item
        mult, why = rel.explain(Item("a lace bralette", tags={"lingerie"}), rel.closure({"beach"}), True)
        self.assertEqual(mult, 0.0)
        self.assertIn("충돌", why)

    def test_trace_is_reported(self):
        r = self.pm.generate("gravure photoshoot on the beach", GenerateOptions(mode="random", seed=3))
        self.assertTrue(any(line.startswith("location:") for line in r.lexicon.get("trace", [])))


class RequestPhraseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pm = PromptMaker(translator=NoTranslator())

    def test_selected_template_and_type_kept_in_full_random(self):
        for seed in range(20):
            r = self.pm.generate("fair skin, white miniskirt", mode="random", prompt_type="image",
                                 template="fashion_editorial", seed=seed)
            self.assertEqual(r.template_id, "fashion_editorial")
        for seed in range(20):
            r = self.pm.generate("fair skin", mode="random", prompt_type="image", seed=seed)
            self.assertEqual(r.prompt_type, "image")

    def test_colour_garment_and_skin_tone_are_used(self):
        for seed in range(30):
            r = self.pm.generate("bright skin, white miniskirt", mode="random", seed=seed)
            p = r.prompt.lower()
            self.assertIn("white mini skirt", p)
            self.assertIn("fair", r.picks.get("skin", "") + p)
            self.assertNotIn("also features", p)
            self.assertNotIn(" women", p)

    def test_tag_dumps_and_minor_ages_are_not_loaded(self):
        from promptmaker.data_loader import is_usable_item, looks_like_tag_list
        self.assertTrue(looks_like_tag_list("Seductress, Short, Firm, Square Face, Fair Skin, Light Brown Hair"))
        self.assertFalse(looks_like_tag_list("with red, gold, and black accents"))
        self.assertFalse(is_usable_item("age", "19"))
        self.assertTrue(is_usable_item("age", "21"))


class ModelProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pm = PromptMaker(translator=NoTranslator())

    def test_defaults_are_krea2_and_minimax_h3(self):
        img = self.pm.generate("portrait", prompt_type="image", seed=1)
        vid = self.pm.generate("walking on the beach", prompt_type="video", seed=1)
        self.assertEqual(img.model_profile, "krea2")
        self.assertEqual(vid.model_profile, "minimax_h3")
        self.assertTrue(vid.prompt.startswith("integrated_multimodal_description: "))
        self.assertIn("\noverall_soundscape: ", vid.prompt)
        self.assertIn("\nnon_diegetic_music: ", vid.prompt)
        self.assertNotIn("cinematic", vid.prompt.lower())

    def test_moss_tags_and_negative(self):
        r = self.pm.generate("white miniskirt", model_profile="moss", seed=2)
        self.assertTrue(r.prompt.startswith("masterpiece, best quality"))
        self.assertNotIn("{", r.prompt)
        self.assertIn("worst quality", r.negative_prompt)
        self.assertNotIn("nsfw", r.negative_prompt)

    def test_qwen_inline_exclusions_and_none(self):
        r = self.pm.generate("portrait", model_profile="qwen2.1", seed=3)
        self.assertTrue(r.prompt.endswith("No text, no watermark, no logos."))
        r2 = self.pm.generate("portrait", model_profile="none", seed=3)
        self.assertEqual(r2.prompt, r2.base_prompt)
        self.assertEqual(r2.model_profile, "")

    def test_old_profiles_removed(self):
        for old in ("flux", "sdxl", "wan_video"):
            self.assertNotIn(old, self.pm.info()["model_profiles"])


class TimeOfDayTests(unittest.TestCase):
    def test_dusk_never_gets_midday_sun(self):
        pm = PromptMaker(translator=NoTranslator())
        for seed in range(60):
            r = pm.generate("beach at dusk", prompt_type="video", seed=seed, model_profile="none")
            low = r.prompt.lower()
            if "dusk" in low or "sunset" in low or "golden hour" in low:
                self.assertNotIn("midday", low, r.prompt)
                self.assertNotIn("noon", low, r.prompt)


class NudeAndFragmentTestCase(unittest.TestCase):
    def test_intent_nude(self):
        from promptmaker import intent as it
        self.assertTrue(it.extract("그라비아 누드 화보", "gravure nude pictorial", {}).nude)
        self.assertTrue(it.extract("a naked woman", "a naked woman", {}).nude)
        self.assertFalse(it.extract("a woman with nude lipstick", "a woman with nude lipstick", {}).nude)
        self.assertFalse(it.extract("a woman in a dress", "a woman in a dress", {}).nude)

    def test_nude_request_has_no_clothes(self):
        from promptmaker import NoTranslator, PromptMaker
        pm = PromptMaker(translator=NoTranslator())
        clothes = ("swimsuit", "bikini", "dress", "shirt", "skirt", "jeans", "blouse", "lingerie", "sweater")
        for seed in range(12):
            res = pm.generate("a beautiful japanese woman, gravure nude photo", seed=seed)
            attire = res.picks.get("attire", "")
            self.assertIn(attire, ("", "nothing at all", "nothing but bare skin"))
            self.assertFalse(any(c in res.prompt.lower() for c in clothes), res.prompt)
            if "attire_state" in res.picks:
                self.assertEqual(res.picks["attire_state"], "fully nude")

    def test_fragment_and_signature_filters(self):
        import re
        from promptmaker import core
        self.assertTrue(core._SUBJECT_LEAD_RE.match("the woman, with dark hair"))
        self.assertTrue(core._SUBJECT_LEAD_RE.match("She is smiling"))
        self.assertFalse(core._SUBJECT_LEAD_RE.match("theatrical stage lights"))
        self.assertFalse(core._SUBJECT_LEAD_RE.match("shoulder-length dark hair"))
        self.assertEqual({m.lower() for m in core._SIGNATURE_RE.findall("in a bedroom with soft Bokeh background")},
                         {"bokeh"})

    def test_bokeh_not_repeated_often(self):
        from promptmaker import NoTranslator, PromptMaker
        pm = PromptMaker(translator=NoTranslator())
        repeated = 0
        for seed in range(30):
            res = pm.generate("a woman in a cafe", seed=seed)
            hits = [k for k, t in res.picks.items() if "bokeh" in str(t).lower()]
            repeated += len(hits) > 1
        self.assertLessEqual(repeated, 2)


class RestrictionTestCase(unittest.TestCase):
    def test_defaults_and_generation(self):
        from promptmaker import NoTranslator, PromptMaker
        pm = PromptMaker(translator=NoTranslator())
        self.assertEqual(pm.restrictions.nationalities, ["Korean", "Japanese"])
        for seed in range(25):
            for req in ("a woman in a cafe", "a 30-year-old Chinese woman on a beach", "an American model"):
                r = pm.generate(req, seed=seed)
                if "nationality" in r.picks and r.picks.get("casting", "").find("man") != 0:
                    self.assertIn(r.picks["nationality"], ("Korean", "Japanese"), req)
                self.assertIn(r.picks["age"], ("20", "21", "22"), req)
                self.assertNotIn("Chinese", r.prompt)
                self.assertNotIn("American", r.prompt)

    def test_user_file_persists(self):
        import tempfile
        from promptmaker import NoTranslator, PromptMaker
        from promptmaker.restrictions import Restrictions
        d = tempfile.mkdtemp()
        Restrictions.from_dict({"nationalities": ["Japanese"], "female_age_min": 21, "female_age_max": 21}).save(d)
        pm = PromptMaker(user_data_dir=d, translator=NoTranslator())
        for seed in range(8):
            r = pm.generate("a woman", seed=seed)
            self.assertEqual(r.picks["age"], "21")
            self.assertEqual(r.picks["nationality"], "Japanese")

    def test_min_age_safety_still_first(self):
        from promptmaker.restrictions import Restrictions
        r = Restrictions.from_dict({"female_age_min": 15, "female_age_max": 18})
        self.assertEqual((r.female_age_min, r.female_age_max), (20, 20))

    def test_apply_to_text_and_allows(self):
        from promptmaker.restrictions import Restrictions
        r = Restrictions()
        out = r.apply_to_text("A photo of a beautiful 25-year-old Chinese woman with a 40-year-old American man.")
        self.assertRegex(out, r"22-year-old (Korean|Japanese) woman")
        self.assertIn("40-year-old American man", out)
        self.assertIn("Korean, Asian woman", r.apply_to_text("a Korean, Asian woman"))
        self.assertTrue(r.allows("nationality", "korean"))
        self.assertFalse(r.allows("nationality", "Thai"))
        self.assertFalse(r.allows("age", "25"))
        self.assertTrue(r.allows("hair", "anything"))

    def test_append_items_skips_restricted(self):
        import os
        import tempfile
        from promptmaker import vision
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "elements"))
        for k in ("nationality", "age"):
            open(os.path.join(d, "elements", k + ".txt"), "w").close()
        rep = vision.append_items(d, {"nationality": ["Thai", "Japanese"], "age": ["25", "21"]})
        self.assertEqual(rep["added"], {"nationality": ["Japanese"], "age": ["21"]})
        self.assertEqual(rep["skipped"]["nationality"][0]["reason"], "restricted")


class RestrictionCleanTestCase(unittest.TestCase):
    def test_clean_with_backup(self):
        import os
        import tempfile
        from promptmaker.restrictions import clean_element_files
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "elements"))
        with open(os.path.join(d, "elements", "nationality.txt"), "w", encoding="utf-8") as f:
            f.write("# head\n3::Korean\nThai\nJapanese\n")
        with open(os.path.join(d, "elements", "age.txt"), "w", encoding="utf-8") as f:
            f.write("20\n2::25\n22\n")
        res = clean_element_files(d)
        self.assertEqual(res["removed"], {"nationality": ["Thai"], "age": ["25"]})
        self.assertTrue(os.path.isfile(os.path.join(res["backup"], "nationality.txt")))
        with open(os.path.join(d, "elements", "nationality.txt"), encoding="utf-8") as f:
            self.assertEqual(f.read(), "# head\n3::Korean\nJapanese\n")
        self.assertEqual(clean_element_files(d)["removed"], {})
