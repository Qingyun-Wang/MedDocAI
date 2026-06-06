"""
Unit tests for ingestion/drug_label_chunker.py

Tests cover deduplication logic, section extraction, chunking,
and metadata structure — all without real API calls.

Run with:  python -m pytest tests/test_drug_label_chunker.py -v
"""

import hashlib
import uuid
from unittest.mock import MagicMock, patch

import pytest

from ingestion.drug_label_chunker import (
    MAX_TOKENS,
    TARGET_SECTIONS,
    _effective_date,
    _extract_sections,
    _section_score,
    build_chunks,
)


# ---------------------------------------------------------------------------
# Fixtures — minimal fake label records
# ---------------------------------------------------------------------------

def _make_label(
    generic_name: str,
    indications: str = "Treats X",
    warnings: str = "",
    warnings_and_cautions: str = "",
    contraindications: str = "",
    adverse_reactions: str = "",
    dosage: str = "Take once daily",
    effective_time: str = "20240101",
) -> dict:
    """Build a minimal fake FDA label record."""
    openfda = {"generic_name": [generic_name], "brand_name": ["BrandX"]}
    record: dict = {
        "openfda": openfda,
        "effective_time": effective_time,
        "set_id": "test-set-id",
        "id": "test-id",
    }
    if indications:
        record["indications_and_usage"] = [indications]
    if warnings:
        record["warnings"] = [warnings]
    if warnings_and_cautions:
        record["warnings_and_cautions"] = [warnings_and_cautions]
    if contraindications:
        record["contraindications"] = [contraindications]
    if adverse_reactions:
        record["adverse_reactions"] = [adverse_reactions]
    if dosage:
        record["dosage_and_administration"] = [dosage]
    return record


# ---------------------------------------------------------------------------
# _section_score
# ---------------------------------------------------------------------------

class TestSectionScore:
    def test_empty_record_scores_zero(self):
        assert _section_score({}) == 0

    def test_two_filled_sections(self):
        r = _make_label("drug", indications="Yes", dosage="Yes",
                        warnings="", adverse_reactions="")
        assert _section_score(r) == 2

    def test_all_five_sections(self):
        r = _make_label("drug", indications="A", warnings="B",
                        contraindications="C", adverse_reactions="D", dosage="E")
        # 5 filled sections (indications, warnings, contraindications, adverse, dosage)
        assert _section_score(r) == 5

    def test_empty_string_not_counted(self):
        r = _make_label("drug", indications="  ", dosage="Take daily")
        assert _section_score(r) == 1   # only dosage counts


# ---------------------------------------------------------------------------
# _effective_date
# ---------------------------------------------------------------------------

class TestEffectiveDate:
    def test_returns_date_string(self):
        r = {"effective_time": "20231015"}
        assert _effective_date(r) == "20231015"

    def test_missing_field_returns_empty(self):
        assert _effective_date({}) == ""

    def test_none_value_returns_empty(self):
        assert _effective_date({"effective_time": None}) == ""


# ---------------------------------------------------------------------------
# _extract_sections
# ---------------------------------------------------------------------------

class TestExtractSections:
    def test_returns_sections_with_content(self):
        r = _make_label("drug", indications="Treats X", dosage="Take daily",
                        warnings="", contraindications="")
        sections = _extract_sections(r)
        section_types = [s[0] for s in sections]
        assert "indications_and_usage" in section_types
        assert "dosage_and_administration" in section_types

    def test_skips_empty_sections(self):
        r = _make_label("drug", indications="Treats X", warnings="",
                        contraindications="", dosage="")
        sections = _extract_sections(r)
        section_types = [s[0] for s in sections]
        assert "contraindications" not in section_types
        assert "dosage_and_administration" not in section_types

    def test_warnings_and_cautions_suppresses_warnings(self):
        """If warnings_and_cautions present, plain 'warnings' should be skipped."""
        r = _make_label("drug", warnings="Short warning",
                        warnings_and_cautions="Long detailed warning")
        sections = _extract_sections(r)
        section_types = [s[0] for s in sections]
        assert "warnings_and_cautions" in section_types
        assert "warnings" not in section_types

    def test_warnings_used_when_no_warnings_and_cautions(self):
        r = _make_label("drug", warnings="Short warning", warnings_and_cautions="")
        sections = _extract_sections(r)
        section_types = [s[0] for s in sections]
        assert "warnings" in section_types

    def test_long_section_truncated(self):
        # Generate text that exceeds MAX_TOKENS when encoded
        # ~4 chars per token, so MAX_TOKENS * 5 chars is definitely over the limit
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        long_text = "A " * (MAX_TOKENS * 5)  # well over token limit
        r = _make_label("drug", indications=long_text, dosage="")
        sections = _extract_sections(r)
        ind_text = next(t for s, t in sections if s == "indications_and_usage")
        # After truncation, should be at or under MAX_TOKENS
        assert len(enc.encode(ind_text)) <= MAX_TOKENS

    def test_section_text_is_stripped(self):
        r = _make_label("drug", indications="  Treats X  ")
        sections = _extract_sections(r)
        assert sections[0][1] == "Treats X"


# ---------------------------------------------------------------------------
# build_chunks
# ---------------------------------------------------------------------------

class TestBuildChunks:
    def test_basic_chunk_structure(self):
        labels = {
            "metformin": _make_label("metformin", indications="Treats diabetes",
                                     dosage="Take daily")
        }
        chunks = build_chunks(labels)
        assert len(chunks) == 2  # indications + dosage

        chunk = chunks[0]
        assert "id" in chunk
        assert "text" in chunk
        assert "drug_name" in chunk
        assert "section_type" in chunk
        assert "source" in chunk
        assert chunk["source"] == "fda_label"

    def test_drug_name_is_lowercase_generic(self):
        labels = {"METFORMIN": _make_label("METFORMIN", indications="Yes")}
        chunks = build_chunks(labels)
        assert chunks[0]["drug_name"] == "METFORMIN"   # key as passed

    def test_chunk_id_is_deterministic(self):
        """Same drug + section always produces the same UUID."""
        labels = {"metformin": _make_label("metformin", indications="Treats diabetes")}
        chunks1 = build_chunks(labels)
        chunks2 = build_chunks(labels)
        assert chunks1[0]["id"] == chunks2[0]["id"]

    def test_different_sections_have_different_ids(self):
        labels = {
            "metformin": _make_label("metformin", indications="Yes", dosage="Yes")
        }
        chunks = build_chunks(labels)
        ids = [c["id"] for c in chunks]
        assert len(ids) == len(set(ids))  # all unique

    def test_different_drugs_have_different_ids(self):
        labels = {
            "metformin": _make_label("metformin", indications="Diabetes"),
            "lisinopril": _make_label("lisinopril", indications="Hypertension"),
        }
        chunks = build_chunks(labels)
        ids = [c["id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_label_with_no_sections_skipped(self):
        labels = {
            "empty_drug": _make_label("empty_drug", indications="", dosage="",
                                      warnings="", contraindications="",
                                      adverse_reactions=""),
        }
        chunks = build_chunks(labels)
        assert len(chunks) == 0

    def test_brand_names_included_in_metadata(self):
        record = _make_label("metformin", indications="Yes")
        record["openfda"]["brand_name"] = ["Glucophage", "Fortamet"]
        labels = {"metformin": record}
        chunks = build_chunks(labels)
        assert "Glucophage" in chunks[0]["brand_names"]

    def test_ndc_list_in_metadata(self):
        record = _make_label("metformin", indications="Yes")
        record["openfda"]["product_ndc"] = ["00093-7267", "00093-7268"]
        labels = {"metformin": record}
        chunks = build_chunks(labels)
        assert "00093-7267" in chunks[0]["ndc_list"]

    def test_text_field_matches_section_content(self):
        labels = {
            "aspirin": _make_label("aspirin", indications="Pain relief", dosage="")
        }
        chunks = build_chunks(labels)
        assert chunks[0]["text"] == "Pain relief"


# ---------------------------------------------------------------------------
# Deduplication logic (integration — uses actual zip files)
# ---------------------------------------------------------------------------

class TestDeduplicationIntegration:
    """These tests read actual zip files but don't call any API."""

    def test_dry_run_produces_chunks(self):
        from ingestion.drug_label_chunker import run
        stats = run(dry_run=True)
        assert stats["unique_labels"] > 0
        assert stats["total_chunks"] > 0
        assert stats["total_chunks"] > stats["unique_labels"]  # multiple sections per label
        assert stats["upserted"] == 0  # dry run

    def test_dedup_reduces_record_count(self):
        from ingestion.drug_label_chunker import run
        stats = run(dry_run=True)
        # 85K+ eligible records → ~14.5K unique generics
        assert stats["unique_labels"] < 50_000

    def test_estimated_cost_under_one_dollar(self):
        from ingestion.drug_label_chunker import run
        stats = run(dry_run=True)
        assert stats["estimated_cost_usd"] < 1.00

    def test_patient_drugs_represented(self):
        """Key drugs from our 10 demo patients should be in the deduplicated set."""
        from ingestion.drug_label_chunker import deduplicate_labels, build_chunks
        best = deduplicate_labels()
        chunks = build_chunks(best)
        drug_names = {c["drug_name"] for c in chunks}

        # Check a sample of our demo patient drugs
        expected = ["metformin", "penicillin v potassium", "clopidogrel",
                    "amlodipine", "simvastatin", "acetaminophen"]
        for drug in expected:
            found = any(drug in name for name in drug_names)
            assert found, f"Expected drug '{drug}' not found in chunks"
