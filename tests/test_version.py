import os
import sys
import unittest
from unittest import mock

import promptmaker
from promptmaker import version

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import github_auto  # noqa: E402


class VersionTests(unittest.TestCase):
    def test_parse_and_compare(self):
        self.assertEqual(version.parse_version("v0.3.1"), (0, 3, 1))
        self.assertEqual(version.parse_version("1.2"), (1, 2))
        self.assertGreater(version.parse_version("v0.10.0"), version.parse_version("0.9.9"))
        with self.assertRaises(ValueError):
            version.parse_version("latest")

    def test_version_files_agree(self):
        self.assertEqual(github_auto.read_version(), promptmaker.__version__)

    def test_bump(self):
        self.assertEqual(github_auto.bump_version("0.2.0", "minor"), "0.3.0")
        self.assertEqual(github_auto.bump_version("0.3.0", "patch"), "0.3.1")
        self.assertEqual(github_auto.bump_version("0.3.4", "major"), "1.0.0")

    def test_check_for_update(self):
        fake = {"tag": "v99.0.0", "version": "99.0.0", "url": "u", "published_at": "", "notes": ""}
        with mock.patch.object(version, "latest_release", return_value=fake):
            info = version.check_for_update()
        self.assertTrue(info["update_available"])
        self.assertIn("@v99.0.0", info["install"])
        same = dict(fake, tag="v" + promptmaker.__version__, version=promptmaker.__version__)
        with mock.patch.object(version, "latest_release", return_value=same):
            self.assertFalse(version.check_for_update()["update_available"])


if __name__ == "__main__":
    unittest.main()
