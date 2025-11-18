import sqlite3
from os import PathLike

import ahocorasick
from classconfig import ConfigurableValue, RelativePathTransformer


def jaccard_similarity(set1: set[str], set2: set[str]) -> float:
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 0.0
    return len(intersection) / len(union)


class SoftwareDatabase:
    """
    Implements a SQLite-based software database.
    """

    db_path: str = ConfigurableValue(
        "Path to the SQLite database file.",
        transform=RelativePathTransformer()
    )
    known_threshold: int = ConfigurableValue(
        "Threshold for considering a software as known based on occurrences.",
        user_default=3,
        validator=lambda x: x >= 0
    )

    def __init__(self, db_path: str | PathLike[str], known_threshold: int = 3):
        """
        Initializes the SQLiteSearchCache with the specified database path.
        """

        self.db_path = db_path
        self.connection = None
        self.cursor = None
        self.automaton = None
        self.known_threshold = known_threshold

    def _add_surface_form_to_automaton(self, surface_form: str, db_id: int):
        ctx_count = self.cursor.execute("SELECT COUNT(*) FROM contexts WHERE software_id = ?", (db_id,)).fetchone()[0]
        if ctx_count >= self.known_threshold:
            self.automaton.add_word(surface_form, surface_form)

    def known_surface_forms_in_text(self, text: str) -> list[tuple[int, str]]:
        """
        Finds all known software surface forms in the given text using Aho-Corasick algorithm.

        :param text: Input text to search for known software names.
        :return: List of found software names in form of (start_index, software_name).
        """
        if self.automaton is None or len(self.automaton) == 0:
            return []

        if self.automaton.kind != ahocorasick.AHOCORASICK:
            self.automaton.make_automaton()

        found = []
        for end_index, software in self.automaton.iter(text.lower()):
            start_index = end_index - len(software) + 1
            if start_index == 0 or end_index == len(text) - 1 or not text[start_index - 1].isalnum() and not text[end_index + 1].isalnum():
                found.append((start_index, software))
        return found

    def exists(self, software: str) -> bool:
        """
        Checks if the software exists in the database.

        :param software: software name
        :return: If exists, returns True, otherwise False.
        """
        software = software.strip().lower()
        self.cursor.execute("SELECT 1 FROM software WHERE name = ?", (software,))
        return self.cursor.fetchone() is not None

    def get_contexts(self, software: str) -> list[str]:
        """
        Gets the contexts for the software.

        :param software: software name
        :return: List of contexts.
        """
        software = software.strip().lower()
        self.cursor.execute("""
            SELECT c.context 
            FROM contexts c
            JOIN software s ON c.software_id = s.id
            WHERE s.name = ?
        """, (software,))
        rows = self.cursor.fetchall()
        return [row[0] for row in rows]

    def get_contexts_sorted(self, software: str, context: str) -> list[str]:
        """
        Gets the contexts for the software, sorted by the most similar to the given context.

        The similarity metric used is the Jaccard similarity between the sets of words in the contexts.
        :param software: software name
        :param context: context string to compare against
        :return: List of contexts sorted by similarity to the given context.
        """
        contexts = self.get_contexts(software)

        if len(contexts) == 0:
            return contexts

        context_set = set(context.lower().split())
        contexts_with_similarity = [
            (ctx, jaccard_similarity(context_set, set(ctx.lower().split()))) for ctx in contexts
        ]
        contexts_with_similarity.sort(key=lambda x: x[1], reverse=True)
        return [ctx for ctx, _ in contexts_with_similarity]

    def add_software(self, software: str, context: str):
        """
        Adds software and its context to the database.

        :param software: software name, it will be stripped and lowercased
        :param context: context string
        :return:
        """
        software = software.strip().lower()
        with self.connection:
            software_id = self.cursor.execute("SELECT id FROM software WHERE name = ?", (software,)).fetchone()
            if software_id is None:
                self.cursor.execute(
                    "INSERT INTO software (name) VALUES (?)",
                    (software,)
                )
                software_id = self.cursor.lastrowid
            else:
                software_id = software_id[0]

            self.cursor.execute("INSERT OR IGNORE INTO contexts (software_id, context) VALUES (?, ?)", (software_id, context))

        if software not in self.automaton:
            self._add_surface_form_to_automaton(software, software_id)

    def __enter__(self):
        if self.connection is not None:
            return self
        self.connection = sqlite3.connect(self.db_path, timeout=10.0)
        self.connection.execute("PRAGMA journal_mode=WAL;")
        self.cursor = self.connection.cursor()

        with self.connection:
            self.cursor.execute(
                "CREATE TABLE IF NOT EXISTS software (id INTEGER PRIMARY KEY, name TEXT UNIQUE)"
            )
            self.cursor.execute(
                "CREATE TABLE IF NOT EXISTS contexts (software_id INTEGER REFERENCES software(id), context TEXT, UNIQUE(software_id, context))")

        self.automaton = ahocorasick.Automaton()

        self.cursor.execute("SELECT id, name FROM software")
        for row in self.cursor.fetchall():
            db_id, name = row
            self._add_surface_form_to_automaton(name, db_id)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None
            self.cursor = None
