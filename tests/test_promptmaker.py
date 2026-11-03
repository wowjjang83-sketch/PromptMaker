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
            for seed in range(15):
                for rating in ("sfw", "suggestive"):
                    r = self.pm.generate("", mode="random_by_element", template=tid,
                                         random_elements=["*"], seed=seed, content_rating=rating)
                    self.assertNotIn("{", r.prompt, (tid, r.prompt))
                    self.assertNotIn("}", r.prompt, (tid, r.prompt))
                    self.assertTrue(r.prompt[0].isupper())
                    self.assertTrue(r.prompt.endswith("."))
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

    def test_rating_filters_items(self):
        suggestive = {i.text for i in self.pm.elements["attire_state"].items if i.rating != "sfw"}
        for seed in range(30):
            r = self.pm.generate("", seed=seed, content_rating="sfw")
            self.assertNotIn(r.picks.get("attire_state"), suggestive)

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
        pm = PromptMaker(translator=DictionaryTranslator(self.pm.ko_en))
        r = pm.generate("25살 한국 여자 영상", seed=1, mode="random")
        self.assertEqual(r.picks["age"], "25")
        self.assertEqual(r.prompt_type, "video")

    def test_user_selection_beats_request(self):
        r = self.pm.generate("a Japanese woman", seed=1, selections={"nationality": "Thai"})
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
                self.assertNotIn(w, p, r.prompt)

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
