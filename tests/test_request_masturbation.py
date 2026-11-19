"""0.13.0 — 요청 내용이 그대로 반영되는지 (AI채팅 오생성 사례) · 자위 동작 테스트."""
import os
import re
import shutil
import tempfile
import unittest

from promptmaker import PromptMaker, masturbation
from promptmaker.element_check import check_item
from promptmaker.intent import extract, place_head
from promptmaker.translator import ko_pre_translate

REQUEST = ("긴생머리에 매혹적인 일본 여성이 혼자서 침대에서 미니슬립 의상을 입고 팬티를 내리며 자위하는 장면의 "
           "그라비아 포트레이트 고화질 사진")
# 구글 번역 결과 (번역 전 처리 후). 테스트는 네트워크 없이 이 번역을 씀
ENGLISH = ("A high-definition gravure portrait of a seductive Japanese woman with long straight hair masturbating "
           "alone in bed wearing a mini slip dress and pulling down her panties.")
# 예전 번역 (번역 전 처리 없이): '생머리' 가 빠지고 'mini-slip costume'
OLD_ENGLISH = ("A high-definition gravure portrait of a seductive Japanese woman with long hair wearing a mini-slip "
               "costume and pulling down her panties while masturbating alone in bed.")

# 사용자 PC 요소 파일에 들어 있던 잘못된 줄 (와일드카드에서 온 것) — 이런 줄이 있어도 요청이 지켜져야 함
JUNK = {
    "location": ["on a gravel garden bed", "on a shallow stream bed", "on a telephone beside bed",
                 "beside her on the bed", "on a window frame above bed", "on a bed with white sheets"],
    "bottom": ["dark shorts and her long", "a long pink-and-red plaid skirt", "white shorts and her hair"],
    "top": ["a sheer black sequined long-sleeve top", "a grey hoodie styling hair"],
    "footwear": ["black high-heeled shoes and her long"],
    "pose": ["standing in a seductive", "looking directly at the camera with a seductive"],
    "hair": ["a neat fringe"],
}


class _Fixed:
    """번역기 대신: 정해진 영어를 돌려줌."""

    def __init__(self, table):
        self.table = table
        self.last_error = None
        self.last_backend = "fixed"

    def translate(self, text):
        return self.table.get(text, text)


def _maker(user_dir=None, table=None):
    return PromptMaker(user_data_dir=user_dir, translator=_Fixed(table or {ko_pre_translate(REQUEST): ENGLISH}))


class RequestReflectedTests(unittest.TestCase):
    """'긴생머리 … 미니슬립 … 팬티를 내리며 자위 … 침대' 요청이 그대로 들어가는지."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(cls.tmp, "elements"))
        for key, lines in JUNK.items():
            with open(os.path.join(cls.tmp, "elements", f"{key}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        cls.makers = {"base": _maker(), "junk": _maker(cls.tmp)}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _check(self, prompt, res):
        low = prompt.lower()
        self.assertIn("mini slip dress", low, prompt)
        self.assertRegex(low, r"straight[ ,a-z-]*hair|hair styled straight", prompt)
        self.assertIn("japanese", low)
        self.assertRegex(prompt, r"\b2[0-2]-year-old\b")
        self.assertRegex(low, r"panties (?:down|and pulling them down)", prompt)
        self.assertTrue(any(re.sub(r"\{her\}", "her", t.split(":", 1)[1].strip().split("  [")[0]) in prompt
                            for t in res.lexicon["trace"] if t.startswith(("masturbation:", "action:", "motion:"))
                            and masturbation.line_kinds(t)), prompt)
        loc = re.search(r"(?:takes place|Set|Background:|showing [^.]*?)\s+((?:on|in|at)[^.,]*)", prompt)
        self.assertTrue(loc and re.search(r"\bbed\b", loc.group(1)), prompt)
        for bad in ("garden", "gravel", "stream", "telephone", "window frame", "standing", "walking", "costume",
                    "the scene also features", "her long,", "and her long", "and her hair", "plaid skirt",
                    "hoodie", "fringe", "high-heeled"):
            self.assertNotIn(bad, low, prompt)

    def test_request_is_reflected(self):
        for name, pm in self.makers.items():
            for seed in range(12):
                with self.subTest(data=name, seed=seed):
                    res = pm.generate(REQUEST, seed=seed)
                    self._check(res.prompt, res)

    def test_old_translation_still_reflected(self):
        """번역 전 처리가 없던 예전 번역 ('long hair', 'mini-slip costume') 에서도 장소·동작·말 조각은 맞아야 함."""
        pm = PromptMaker(user_data_dir=self.tmp, translator=_Fixed({ko_pre_translate(REQUEST): OLD_ENGLISH}))
        for seed in range(8):
            p = pm.generate(REQUEST, seed=seed).prompt.lower()
            self.assertNotIn("the scene also features", p)
            self.assertNotIn("gravel", p)
            self.assertNotIn("standing", p)
            self.assertRegex(p, r"panties (?:down|and pulling them down)")
            self.assertRegex(p, r"long[ ,a-z-]*hair")

    def test_intent(self):
        pm = self.makers["junk"]
        it = extract(REQUEST, ENGLISH, pm.elements)
        self.assertTrue(it.masturbation and it.undressing and not it.nude)
        self.assertEqual(it.locks.get("nationality"), "Japanese")
        self.assertTrue(all(place_head(t) == "bed" for t in it.pools["location"]), it.pools["location"])
        self.assertNotIn("pose", it.pools)  # 'seductive' 로 포즈가 잠기지 않음
        self.assertNotIn("bottom", it.pools)  # 'long' 으로 긴 치마가 잠기지 않음
        self.assertTrue(all("straight" in h and "long" in h for h in it.pools.get("hair", [it.locks.get("hair", "")])))


class MasturbationTests(unittest.TestCase):
    def test_detect_korean(self):
        self.assertEqual(masturbation.detect("침대에서 자위하는 여성"), (True, []))
        self.assertEqual(masturbation.detect("가슴을 주무르며 자위")[1], ["breast"])
        self.assertEqual(masturbation.detect("클리토리스를 만지는 여성")[1], ["clit"])
        self.assertEqual(masturbation.detect("질에 손가락을 넣고 자위")[1], ["finger"])
        self.assertFalse(masturbation.detect("가슴이 큰 여성")[0])
        self.assertFalse(masturbation.detect("손가락으로 브이")[0])

    def test_detect_english(self):
        self.assertTrue(masturbation.detect("", "a woman masturbating on the sofa")[0])
        self.assertEqual(masturbation.detect("", "a woman fondling her breasts")[1], ["breast"])
        self.assertEqual(masturbation.detect("", "touching herself, rubbing her clit")[1], ["clit"])
        self.assertFalse(masturbation.detect("", "a woman touching her hair")[0])

    def _lines(self, request, seeds=6):
        pm = PromptMaker(translator=_Fixed({}))
        out = []
        for s in range(seeds):
            res = pm.generate(request, seed=s)
            got = [t.split(":", 1)[1].split("  [")[0].strip() for t in res.lexicon["trace"]
                   if t.startswith(("masturbation:", "action:", "motion:")) and masturbation.line_kinds(t)]
            self.assertTrue(got, res.prompt)
            out.append((res.prompt, got[0]))
        return out

    def test_requested_kind_is_used(self):
        for req, kind in (("a Korean woman masturbating in bed, fondling her breasts", "breast"),
                          ("a Japanese woman masturbating on the sofa, rubbing her clit", "clit"),
                          ("a woman masturbating with her fingers inside her vagina", "finger")):
            for prompt, line in self._lines(req):
                self.assertEqual(masturbation.line_kinds(line), {kind}, (req, prompt))
                self.assertNotRegex(prompt, r"\b(?:standing|walking|dancing)\b")

    def test_scene_is_private_and_solo(self):
        for prompt, _ in self._lines("a Korean woman masturbating", seeds=8):
            self.assertNotRegex(prompt.lower(), r"\b(?:he|his|man|boyfriend)\b")
            self.assertNotRegex(prompt.lower(), r"\b(?:sneakers|high heels|boots|street|market|park)\b")

    def test_location_request_kept(self):
        pm = PromptMaker(translator=_Fixed({}))
        for s in range(4):
            self.assertIn("on a sofa", pm.generate("a woman masturbating on the sofa", seed=s).prompt)

    def test_panties_only_when_asked(self):
        pm = PromptMaker(translator=_Fixed({}))
        for s in range(10):
            self.assertNotIn("panties", pm.generate("a woman taking off her clothes in her bedroom", seed=s).prompt)


class HelperTests(unittest.TestCase):
    def test_place_head(self):
        self.assertEqual(place_head("on a bed with white sheets"), "bed")
        self.assertEqual(place_head("on a gravel garden bed"), "garden bed")
        self.assertEqual(place_head("on a shallow stream bed"), "stream bed")
        self.assertEqual(place_head("on a telephone beside bed"), "telephone")
        self.assertEqual(place_head("beside the bed"), "")
        self.assertEqual(place_head("in a cozy cafe with wooden interiors"), "cafe")

    def test_ko_pre_translate(self):
        self.assertIn("long straight hair", ko_pre_translate("긴생머리 여자"))
        self.assertIn("mini slip dress", ko_pre_translate("미니슬립 의상을 입고"))
        self.assertNotIn("의상", ko_pre_translate("미니슬립 의상을 입고"))

    def test_junk_lines(self):
        self.assertEqual(check_item("bottom", "dark shorts and her long")["text"], "dark shorts")
        self.assertEqual(check_item("top", "a white top with thin straps and her hair")["text"],
                         "a white top with thin straps")
        self.assertEqual(check_item("pose", "standing in a seductive")["status"], "fragment")
        self.assertEqual(check_item("lighting", "flattering glow. - 她的手")["status"], "fragment")
        self.assertEqual(check_item("dress", "a silk mini slip dress")["status"], "ok")


if __name__ == "__main__":
    unittest.main()


class MiniDressRequestTest(unittest.TestCase):
    """0.15.2: '미니원피스 · 미니드레스' 요청은 미니 원피스로 (전엔 아무 원피스 + 'features mini')."""

    def test_korean_terms(self):
        from promptmaker.translator import ko_pre_translate, normalize_english
        for ko in ("미니원피스", "미니 원피스", "미니드레스", "미니 드레스"):
            self.assertEqual(ko_pre_translate(ko), "mini dress")
        self.assertEqual(normalize_english("a red mini-dress"), "a red mini dress")
        self.assertEqual(normalize_english("a mini one-piece"), "a mini dress")

    def test_generation_keeps_mini_dress(self):
        from promptmaker import NoTranslator, PromptMaker
        maker = PromptMaker(translator=NoTranslator())
        for _ in range(5):
            res = maker.generate("a woman wearing a mini dress on a bed")
            text = getattr(res, "prompt", None) or res["prompt"]
            self.assertIn("mini", text.lower())
            self.assertRegex(text.lower(), r"mini (?:slip )?dress")
            self.assertNotIn("features mini", text)
