import shutil
import unittest
from pathlib import Path

from sofairagent.software_database import SoftwareDatabase

SCRIPT_DIR = Path(__file__).parent
TMP_DIR = SCRIPT_DIR / "tmp"

DB_PATH = TMP_DIR / "test.db"


def create_tmp():
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True, exist_ok=True)


class TestSoftwareDatabase(unittest.TestCase):
    def setUp(self):
        create_tmp()
        self.db = SoftwareDatabase(DB_PATH, known_threshold=1)
        self.db.__enter__()

    def tearDown(self):
        self.db.__exit__(None, None, None)
        if TMP_DIR.exists():
            shutil.rmtree(TMP_DIR)

    def exists_empty(self):
        self.assertFalse(self.db.exists("nonexistent_software"))

    def test_get_contexts_empty(self):
        contexts = self.db.get_contexts("nonexistent_software")
        self.assertEqual(len(contexts), 0)

    def test_get_known_from_text_empty(self):
        found = self.db.known_surface_forms_in_text("This text does not contain any known software.")
        self.assertEqual(len(found), 0)

    def test_add_and_exists(self):
        self.db.add_software("TestSoftware", "This is a test context.")
        self.assertTrue(self.db.exists("TestSoftware"))
        self.assertTrue(self.db.exists("testsoftware"))
        self.assertFalse(self.db.exists("AnotherSoftware"))

        #check contexts
        contexts = self.db.get_contexts("TestSoftware")
        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0], "This is a test context.")

    def test_add_multiple_contexts(self):
        self.db.add_software("MultiContextSoftware", "First context.")
        self.db.add_software("MultiContextSoftware", "Second context.")

        contexts = self.db.get_contexts("MultiContextSoftware")
        self.assertEqual(len(contexts), 2)
        self.assertIn("First context.", contexts)
        self.assertIn("Second context.", contexts)

    def test_get_contexts_sorted(self):
        self.db.add_software("SortedSoftware", "Context B about algorithms.")
        self.db.add_software("SortedSoftware", "Context C about data structures.")
        self.db.add_software("SortedSoftware", "Context A about sorting algorithms.")

        sorted_contexts = self.db.get_contexts_sorted("SortedSoftware", "sorting algorithms")
        self.assertEqual(len(sorted_contexts), 3)
        self.assertSequenceEqual(
            ["Context A about sorting algorithms.", "Context B about algorithms.", "Context C about data structures."], sorted_contexts
        )

    def test_get_known_from_text(self):
        self.db.add_software("AlphaSoft", "Context about AlphaSoft.")
        self.db.add_software("BetaTool", "Context about BetaTool.")
        self.db.add_software("GammaApp", "Context about GammaApp.")

        text = "This text mentions AlphaSoft and GammaApp but not the other one."
        found = self.db.known_surface_forms_in_text(text)
        self.assertEqual(len(found), 2)
        self.assertIn((19, "alphasoft"), found)
        self.assertIn((33, "gammaapp"), found)
        text = "BetaTool"
        found = self.db.known_surface_forms_in_text(text)
        self.assertEqual(len(found), 1)
        self.assertIn((0, "betatool"), found)

        text = "BetaTool is great."
        found = self.db.known_surface_forms_in_text(text)
        self.assertEqual(len(found), 1)
        self.assertIn((0, "betatool"), found)

        text = "I know about Betatool"
        found = self.db.known_surface_forms_in_text(text)
        self.assertEqual(len(found), 1)
        self.assertIn((13, "betatool"), found)

        text = "I know about Betatool!"
        found = self.db.known_surface_forms_in_text(text)
        self.assertEqual(len(found), 1)
        self.assertIn((13, "betatool"), found)

        text = "This text mentions desBetatoolose."
        found = self.db.known_surface_forms_in_text(text)
        self.assertEqual(len(found), 0)

    def test_get_known_from_text_threshold_2(self):
        self.db.known_threshold = 2

        self.db.add_software("AlphaSoft", "Context about AlphaSoft.")
        self.db.add_software("BetaTool", "Context about BetaTool.")
        self.db.add_software("BetaTool", "Another context about BetaTool.")
        self.db.add_software("GammaApp", "Context about GammaApp.")

        text = "This text mentions AlphaSoft and BetaTool but not the other one."
        found = self.db.known_surface_forms_in_text(text)
        self.assertEqual(len(found), 1)
        self.assertIn((33, "betatool"), found)




