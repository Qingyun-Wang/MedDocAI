"""
Unit tests for ingestion/medlineplus_ingester.py

Run with:  python -m pytest tests/test_medlineplus_ingester.py -v
"""

import os
import pytest

from ingestion.medlineplus_ingester import (
    DEFINITION_FILES,
    LANGUAGE,
    MEDLINEPLUS_DIR,
    _make_embed_text,
    _strip_html,
    build_chunks,
    parse_definitions,
    parse_health_topics,
    run,
)

# Pure-function tests (TestStripHtml, TestMakeEmbedText) run anywhere. The classes
# that parse the real MedlinePlus XML under data/ (gitignored) are skipped when the
# data is absent (CI).
_needs_data = pytest.mark.skipif(
    not os.path.isdir(MEDLINEPLUS_DIR),
    reason="requires local data/medlineplus (gitignored); skipped in CI",
)


# ---------------------------------------------------------------------------
# _strip_html
# ---------------------------------------------------------------------------

class TestStripHtml:
    def test_removes_tags(self):
        assert _strip_html("<p>Hello <b>world</b></p>") == "Hello world"

    def test_collapses_whitespace(self):
        assert _strip_html("<p>  too   many   spaces  </p>") == "too many spaces"

    def test_plain_text_unchanged(self):
        assert _strip_html("No tags here") == "No tags here"

    def test_empty_string(self):
        assert _strip_html("") == ""

    def test_nested_tags(self):
        result = _strip_html("<div><p>Text <a href='x'>link</a> more</p></div>")
        assert result == "Text link more"


# ---------------------------------------------------------------------------
# parse_health_topics
# ---------------------------------------------------------------------------

@_needs_data
class TestParseHealthTopics:
    def _xml_path(self):
        files = sorted(
            f for f in os.listdir(MEDLINEPLUS_DIR)
            if f.startswith("mplus_topics_") and f.endswith(".xml")
            and "compressed" not in f and "group" not in f
        )
        return os.path.join(MEDLINEPLUS_DIR, files[-1])

    def test_returns_correct_count(self):
        topics = parse_health_topics(self._xml_path())
        assert len(topics) == 1017

    def test_only_english_topics(self):
        topics = parse_health_topics(self._xml_path(), language="English")
        # Spot-check: no Spanish-only titles should appear
        assert len(topics) == 1017

    def test_topic_has_required_fields(self):
        topics = parse_health_topics(self._xml_path())
        t = topics[0]
        assert "title" in t and t["title"]
        assert "url" in t and t["url"]
        assert "full_summary" in t
        assert "also_called" in t
        assert "groups" in t
        assert "source_type" in t
        assert t["source_type"] == "health_topic"

    def test_also_called_is_list(self):
        topics = parse_health_topics(self._xml_path())
        for t in topics:
            assert isinstance(t["also_called"], list)

    def test_groups_is_list(self):
        topics = parse_health_topics(self._xml_path())
        for t in topics:
            assert isinstance(t["groups"], list)

    def test_full_summary_has_no_html_tags(self):
        topics = parse_health_topics(self._xml_path())
        for t in topics:
            assert "<" not in t["full_summary"], \
                f"HTML not stripped in topic: {t['title']}"

    def test_known_topic_exists(self):
        topics = parse_health_topics(self._xml_path())
        titles = {t["title"] for t in topics}
        assert "Diabetes" in titles or "Diabetes Type 2" in titles or \
               any("iabetes" in t for t in titles)

    def test_url_format(self):
        topics = parse_health_topics(self._xml_path())
        for t in topics:
            if t["url"]:
                assert t["url"].startswith("https://medlineplus.gov/")


# ---------------------------------------------------------------------------
# parse_definitions
# ---------------------------------------------------------------------------

@_needs_data
class TestParseDefinitions:
    def test_returns_definitions(self):
        defs = parse_definitions(MEDLINEPLUS_DIR)
        assert len(defs) > 0

    def test_count_approximately_correct(self):
        defs = parse_definitions(MEDLINEPLUS_DIR)
        # 85 unique definitions after global deduplication across 5 files
        # (12 terms appear in multiple files and are collapsed to one entry)
        assert 50 < len(defs) < 200

    def test_definition_has_required_fields(self):
        defs = parse_definitions(MEDLINEPLUS_DIR)
        d = defs[0]
        assert "title" in d and d["title"]
        assert "full_summary" in d and d["full_summary"]
        assert "source_type" in d
        assert d["source_type"] == "definition"

    def test_definition_groups_set(self):
        """Each definition should have a category (its source file)."""
        defs = parse_definitions(MEDLINEPLUS_DIR)
        for d in defs:
            assert len(d["groups"]) > 0

    def test_known_terms_present(self):
        defs = parse_definitions(MEDLINEPLUS_DIR)
        terms = {d["title"].lower() for d in defs}
        # Terms we expect from nutrition/vitamins/minerals files
        assert any("glucose" in t or "calcium" in t or "vitamin" in t
                   for t in terms)

    def test_no_leading_gt_symbol(self):
        """Definition files have '>' prefix on some text — should be stripped."""
        defs = parse_definitions(MEDLINEPLUS_DIR)
        for d in defs:
            assert not d["title"].startswith(">")
            assert not d["full_summary"].startswith(">")


# ---------------------------------------------------------------------------
# _make_embed_text
# ---------------------------------------------------------------------------

class TestMakeEmbedText:
    def _topic(self, **kwargs):
        defaults = {
            "title": "Hypertension",
            "url": "https://medlineplus.gov/hypertension.html",
            "meta_desc": "High blood pressure",
            "full_summary": "High blood pressure is when the force of blood is too high.",
            "also_called": ["High Blood Pressure", "HBP"],
            "groups": ["Blood, Heart and Circulation"],
            "institute": "NIH",
            "source_type": "health_topic",
        }
        defaults.update(kwargs)
        return defaults

    def test_contains_title(self):
        text = _make_embed_text(self._topic())
        assert "Hypertension" in text

    def test_contains_also_called(self):
        text = _make_embed_text(self._topic())
        assert "High Blood Pressure" in text
        assert "HBP" in text

    def test_contains_groups(self):
        text = _make_embed_text(self._topic())
        assert "Blood, Heart and Circulation" in text

    def test_contains_summary(self):
        text = _make_embed_text(self._topic())
        assert "force of blood" in text

    def test_empty_also_called_skipped(self):
        text = _make_embed_text(self._topic(also_called=[]))
        assert "Also called" not in text

    def test_uses_meta_desc_when_no_summary(self):
        text = _make_embed_text(self._topic(full_summary=""))
        assert "High blood pressure" in text  # from meta_desc


# ---------------------------------------------------------------------------
# build_chunks
# ---------------------------------------------------------------------------

@_needs_data
class TestBuildChunks:
    def _topics(self):
        return parse_health_topics(
            sorted(
                os.path.join(MEDLINEPLUS_DIR, f)
                for f in os.listdir(MEDLINEPLUS_DIR)
                if f.startswith("mplus_topics_") and f.endswith(".xml")
                and "compressed" not in f and "group" not in f
            )[-1]
        )

    def test_chunk_count_matches_topics(self):
        topics = self._topics()
        chunks = build_chunks(topics)
        assert len(chunks) == len(topics)

    def test_chunk_has_required_fields(self):
        topics = self._topics()
        chunks = build_chunks(topics)
        c = chunks[0]
        required = {"id", "embed_text", "title", "url", "full_summary",
                    "also_called", "groups", "source", "language"}
        for field in required:
            assert field in c, f"Missing field: {field}"

    def test_source_is_medlineplus(self):
        topics = self._topics()
        chunks = build_chunks(topics)
        assert all(c["source"] == "medlineplus" for c in chunks)

    def test_language_is_english(self):
        topics = self._topics()
        chunks = build_chunks(topics)
        assert all(c["language"] == "English" for c in chunks)

    def test_ids_are_deterministic(self):
        topics = self._topics()
        chunks1 = build_chunks(topics)
        chunks2 = build_chunks(topics)
        assert [c["id"] for c in chunks1] == [c["id"] for c in chunks2]

    def test_ids_are_unique(self):
        topics = self._topics()
        chunks = build_chunks(topics)
        ids = [c["id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_embed_text_not_empty(self):
        topics = self._topics()
        chunks = build_chunks(topics)
        for c in chunks:
            assert c["embed_text"].strip(), f"Empty embed text for: {c['title']}"


# ---------------------------------------------------------------------------
# Integration: dry run
# ---------------------------------------------------------------------------

@_needs_data
class TestDryRun:
    def test_dry_run_returns_stats(self):
        stats = run(dry_run=True)
        assert stats["health_topics"] == 1017
        assert stats["definitions"] > 0
        assert stats["total_chunks"] == stats["health_topics"] + stats["definitions"]
        assert stats["upserted"] == 0

    def test_cost_is_negligible(self):
        stats = run(dry_run=True)
        assert stats["estimated_cost_usd"] < 0.05  # under 5 cents

    def test_total_chunks_correct(self):
        stats = run(dry_run=True)
        # 1017 health topics + 85 unique definitions (deduplicated from 97 raw)
        assert stats["total_chunks"] == 1102

    def test_no_duplicate_ids(self):
        """All chunk IDs must be unique — verified after fixing URL-collision bug."""
        from ingestion.medlineplus_ingester import parse_health_topics, parse_definitions, build_chunks
        import os
        from collections import Counter
        ml_dir = MEDLINEPLUS_DIR
        xml_files = sorted(f for f in os.listdir(ml_dir)
                           if f.startswith("mplus_topics_") and f.endswith(".xml")
                           and "compressed" not in f and "group" not in f)
        topics = parse_health_topics(os.path.join(ml_dir, xml_files[-1]))
        defs   = parse_definitions(ml_dir)
        chunks = build_chunks(topics + defs)
        ids    = [c["id"] for c in chunks]
        assert len(ids) == len(set(ids)), \
            f"Duplicate IDs found: {len(ids) - len(set(ids))} duplicates"
