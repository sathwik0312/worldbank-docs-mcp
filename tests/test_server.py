"""Offline tests: the World Bank API helper `_call` is replaced by a fake, so nothing touches the network.

Tools are called through FastMCP's in-memory Client, so argument schemas and validation run for real.
Result truncation lives in the client (client/agent.py fit()), so it is tested here too.
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest
from fastmcp import Client

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "server"), str(ROOT / "client")]
import agent  # noqa: E402
import wb_docs_mcp as wb  # noqa: E402


class FakeAPI:
    """Stands in for wb._call. Records every request; knows a few countries for the live-country check."""

    def __init__(self, countries=("Kenya",), n_keywords=0):
        self.calls, self.countries = [], set(countries)
        # Raw API shape: country is "count", keywords are {"0": {"keywd": ...}, "1": ...}.
        self.doc = {"id": "1", "display_title": "A document", "docdt": "2020-01-01T00:00:00Z", "count": "Kenya",
                    "pdfurl": "http://example.org/1.pdf",
                    "keywd": {str(i): {"keywd": f" kw{i} "} for i in range(n_keywords)}}  # API pads some

    async def __call__(self, params):
        self.calls.append(params)
        if set(params) == {"count_exact", "rows"} and params["rows"] == 0:  # _is_country lookup
            return {"total": 7 if params["count_exact"] in self.countries else 0, "documents": {"facets": {}}}
        return {"total": 1, "documents": {"D1": self.doc, "facets": {}}}

    def searches(self):
        return [c for c in self.calls if "fl" in c]  # document searches, not country lookups


@pytest.fixture
def api(monkeypatch):
    fake = FakeAPI()
    monkeypatch.setattr(wb, "_call", fake)
    return fake


def call(tool, **args):
    async def go():
        async with Client(wb.mcp) as client:
            return (await client.call_tool(tool, args)).data
    return asyncio.run(go())


# ---------- whole-document truncation (client fit) ----------

def page(n_docs, offset=0, note=None, title_len=900):
    data = {"total": 500, "offset": offset, "returned": n_docs, "next_offset": offset + n_docs,
            "documents": [{"id": str(i), "display_title": f"{i}-" + "t" * title_len} for i in range(n_docs)]}
    if note:
        data["note"] = note
    return data


def test_fit_drops_whole_documents_and_adds_offset_note():
    original = page(20, offset=40)
    out = agent.fit(json.dumps(original))
    data = json.loads(out)  # never broken JSON
    n = data["returned"]
    assert len(out) <= agent.MAX_RESULT_CHARS
    assert 0 < n < 20 and len(data["documents"]) == n
    assert data["documents"] == original["documents"][:n]  # kept documents are untouched, from the front
    assert data["next_offset"] == 40 + n
    assert data["note"] == f"Showing {n} of 20 documents; call again with offset={40 + n} for more."


def test_fit_keeps_existing_note():
    out = json.loads(agent.fit(json.dumps(page(20, note="Note: 'Kenya' is a country; searched by country."))))
    assert out["note"].startswith("Note: 'Kenya' is a country; searched by country. Showing ")


def test_fit_leaves_small_results_alone_and_replaces_oversized_non_pages():
    small = json.dumps(page(2, title_len=10))
    assert agent.fit(small) == small
    assert "message" in json.loads(agent.fit(json.dumps({"keywords": "k" * 20000})))
    assert "message" in json.loads(agent.fit("x" * 20000))


# ---------- region <-> country fallback ----------

def test_region_given_as_country_is_searched_by_region(api):
    result = call("filter_documents", country="Sub-Saharan Africa")
    assert result["note"] == "Note: 'Sub-Saharan Africa' is a region; searched by region."
    (search,) = api.searches()
    assert search["admreg_exact"] == wb.REGION_ALIASES["Sub-Saharan Africa"]
    assert "count_exact" not in search


def test_country_given_as_region_is_searched_by_country(api):
    result = call("filter_documents", region="Kenya")
    assert result["note"] == "Note: 'Kenya' is a country; searched by country."
    (search,) = api.searches()
    assert search["count_exact"] == "Kenya" and "admreg_exact" not in search


def test_unknown_region_is_left_alone(api):
    result = call("filter_documents", region="Atlantis")
    assert "note" not in result
    assert api.searches()[0]["admreg_exact"] == "Atlantis"


def test_real_region_is_not_checked_as_country(api):
    call("filter_documents", region="South Asia")
    assert api.calls == api.searches()  # no count_exact lookup was made


# ---------- output shape ----------

def test_results_call_the_country_field_country(api):
    (doc,) = call("search_documents", query="water")["documents"]
    assert doc["country"] == "Kenya" and "count" not in doc
    # Renamed in place: moving it after pdfurl cost Q2 all its PDF links (README, API findings).
    assert list(doc) == ["id", "display_title", "docdt", "country", "pdfurl"]


@pytest.mark.parametrize("n_keywords, kept, truncated", [(30, 20, True), (5, 5, False)])
def test_get_document_caps_keywords_at_20(monkeypatch, n_keywords, kept, truncated):
    monkeypatch.setattr(wb, "_call", FakeAPI(n_keywords=n_keywords))
    doc = call("get_document", document_id="1")
    assert doc["keywd"] == [f"kw{i}" for i in range(kept)]  # the first ones, in order, stripped
    assert doc.get("keywd_truncated", False) is truncated


def test_get_document_renames_count_to_country_in_place(api):
    doc = call("get_document", document_id="1")
    assert doc["country"] == "Kenya" and "count" not in doc
    assert list(doc) == ["id", "display_title", "docdt", "country", "pdfurl", "keywd"]  # same position


# ---------- validation hints (returned as results, API never called) ----------

def test_bad_date_format_returns_hint(api):
    result = call("filter_documents", country="Kenya", start_date="Jan 2020")
    assert result["message"].startswith("Invalid arguments. start_date:")
    assert result["message"].endswith("Fix them and call again.")
    assert api.calls == []


def test_reversed_date_range_returns_hint(api):
    result = call("filter_documents", country="Kenya", start_date="2022-01-01", end_date="2019-01-01")
    assert "start_date must be on or before end_date" in result["message"]
    assert api.calls == []


def test_no_filters_returns_hint(api):
    result = call("filter_documents", query="water")
    assert "Give at least one filter" in result["message"]
    assert "use search_documents" in result["message"]
    assert api.calls == []


def test_unknown_facet_field_returns_hint(api):
    result = call("get_facets", field="colour")
    assert result["message"].startswith("Invalid arguments. field:")
    for field in wb.FIELDS:
        assert repr(field) in result["message"]
    assert api.calls == []
