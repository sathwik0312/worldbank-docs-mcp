"""MCP server wrapping the World Bank Documents & Reports API (https://search.worldbank.org/api/v3/wds).

Run (stdio):   python server/wb_docs_mcp.py
Inspect:       npx @modelcontextprotocol/inspector .venv/bin/python server/wb_docs_mcp.py

API facts verified against the live API (Nov 2026):
- Documents come back under data["documents"] keyed "D<id>", next to a "facets" key.
- `topic_exact` matches nothing; the real topic field is `teratopic_exact`.
- "Economic & Sector Work" etc. are `majdocty_exact` values, not `docty_exact`.
- Region is `admreg_exact`. "Sub-Saharan Africa" matches 0; it is split into
  "Eastern and Southern Africa" + "Western and Central Africa" (plus legacy "Africa"). `^` ORs values.
- *_exact filters are case-sensitive ("Kenya" matches, "kenya" does not).
- Facets return at most the top 100 values by count.
- The API calls the country field `count`; every tool result renames it to `country`.
- Date sorting is `srt=docdt&order=desc` (`sort=` is silently ignored).
- Malformed dates make the API return HTTP 500, so dates are validated before calling.
- Offsets above 100000 make the API return HTTP 400.

Design for small models: flat tool arguments; validation problems and empty results come back as
normal results with a one-line "message" telling the model how to retry. Only API/network failures raise.
"""

from datetime import date
from typing import Annotated, Any, Literal

import httpx
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field, ValidationError, model_validator

API_URL = "https://search.worldbank.org/api/v3/wds"
LIST_FIELDS = ["id", "display_title", "docdt", "docty", "majdocty", "count", "pdfurl"]
ABSTRACT_CHARS = 300
KEYWD_MAX = 20  # get_document keeps the first 20 keywords

# Friendly name the LLM sees -> API field. Shared by filter_documents and get_facets.
FIELDS = {
    "country": "count_exact",
    "language": "lang_exact",
    "document_type": "docty_exact",
    "major_document_type": "majdocty_exact",
    "topic": "teratopic_exact",
    "theme": "theme_exact",
    "region": "admreg_exact",
}
FacetField = Literal[tuple(FIELDS)]
REGION_ALIASES = {"Sub-Saharan Africa": "Eastern and Southern Africa^Western and Central Africa^Africa"}
# admreg_exact facet values (live API, Nov 2026), minus "Other". Used to catch regions passed as `country`.
REGIONS = {
    "Eastern and Southern Africa", "Western and Central Africa", "East Asia and Pacific",
    "Latin America and Caribbean", "Europe and Central Asia", "South Asia",
    "Middle East, North Africa, Afghanistan, and Pakistan", "Africa", "Northern Africa",
} | set(REGION_ALIASES)

mcp = FastMCP(
    "worldbank-docs",
    instructions=(
        "Search World Bank publications, reports and project documents. Filter values are exact and "
        "case-sensitive: call get_facets to find them before filter_documents. If a result has a "
        "'message', follow it and call again."
    ),
)


# ---------- shared helpers ----------

async def _call(params: dict[str, Any]) -> dict:
    """GET the WDS API with format=json; turn every failure into a readable ToolError."""
    query = {"format": "json", **{k: v for k, v in params.items() if v is not None}}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(API_URL, params=query)
            r.raise_for_status()
            return r.json()
    except httpx.TimeoutException:
        raise ToolError("World Bank API timed out after 30s. Try again or narrow the query.")
    except httpx.HTTPStatusError as e:
        raise ToolError(f"World Bank API returned HTTP {e.response.status_code}. Try again with simpler arguments.")
    except httpx.HTTPError as e:
        raise ToolError(f"Could not reach the World Bank API: {e}")
    except ValueError:
        raise ToolError("World Bank API returned a non-JSON response.")


def _validate(model: type[BaseModel], **kwargs):
    """Build the input model. Returns (model, None) or (None, {"message": ...}) the LLM can act on."""
    try:
        return model(**kwargs), None
    except ValidationError as e:
        problems = []
        for err in e.errors():
            msg = err["msg"].removeprefix("Value error, ").rstrip(".") + "."
            problems.append(f"{err['loc'][0]}: {msg}" if err["loc"] else msg)
        return None, {"message": "Invalid arguments. " + " ".join(problems) + " Fix them and call again."}


def _clean(value: Any) -> Any:
    """Flatten the API's odd nesting: {"0": {"x": a}, "1": {"x": b}} -> [a, b], {"cdata!": t} -> t."""
    if isinstance(value, dict):
        if "cdata!" in value:
            return value["cdata!"]
        if value and all(k.isdigit() for k in value):
            return [_clean(next(iter(v.values()))) if isinstance(v, dict) and len(v) == 1 else _clean(v)
                    for v in value.values()]
        return {k: _clean(v) for k, v in value.items()}
    return value


def _documents(data: dict) -> list[dict]:
    return [_clean(doc) for key, doc in data.get("documents", {}).items() if key != "facets"]


def _summary(doc: dict) -> dict:
    # The API's country field is "count"; renamed in place ("count" next to total/returned reads like a number).
    out = {("country" if k == "count" else k): doc[k] for k in LIST_FIELDS if k in doc}
    abstract = doc.get("abstracts")
    if isinstance(abstract, str) and abstract:
        out["abstract"] = abstract[:ABSTRACT_CHARS] + ("..." if len(abstract) > ABSTRACT_CHARS else "")
    return out


def _page(data: dict, offset: int, rows: int, empty_hint: str) -> dict:
    docs = [_summary(d) for d in _documents(data)]
    total = int(data.get("total", 0))
    result = {
        "total": total,
        "offset": offset,
        "returned": len(docs),
        "next_offset": offset + rows if offset + rows < total else None,
        "documents": docs,
    }
    if total == 0:
        result["message"] = empty_hint
    elif not docs:
        result["message"] = f"offset {offset} is past the last result (total {total}). Use a smaller offset."
    return result


async def _is_country(name: str) -> bool:
    return int((await _call({"count_exact": name, "rows": 0})).get("total", 0)) > 0


# ---------- input models (constraints live here; descriptions live on the tool args) ----------

class Input(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def _blank_is_omitted(cls, data: Any) -> Any:
        """Inspector and small models send "" for an empty optional argument: treat it as omitted.
        Only fields defaulting to None; required ones (query, document_id, field) still reject ""."""
        if not isinstance(data, dict):
            return data
        optional = {k for k, f in cls.model_fields.items() if f.default is None}
        return {k: None if k in optional and isinstance(v, str) and not v.strip() else v for k, v in data.items()}


class Page(Input):
    rows: int = Field(10, ge=1, le=20)
    offset: int = Field(0, ge=0, le=100_000)  # API returns HTTP 400 above this
    newest_first: bool = False

    def paging(self) -> dict:
        """rows/os/fl (+ newest-first sort) params shared by search_documents and filter_documents."""
        params = {"rows": self.rows, "os": self.offset, "fl": ",".join(LIST_FIELDS + ["abstracts"])}
        return params | ({"srt": "docdt", "order": "desc"} if self.newest_first else {})


class SearchInput(Page):
    query: str = Field(..., min_length=1)
    language: str | None = None


class FilterInput(Page):
    country: str | None = None
    region: str | None = None
    topic: str | None = None
    document_type: str | None = None
    major_document_type: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    query: str | None = None

    @model_validator(mode="after")
    def _check(self):
        if not any([self.country, self.region, self.topic, self.document_type, self.major_document_type,
                    self.start_date, self.end_date]):
            raise ValueError("Give at least one filter (country, region, topic, document_type, "
                             "major_document_type, start_date or end_date), or use search_documents for keywords only.")
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must be on or before end_date.")
        return self


class DocumentInput(Input):
    document_id: str = Field(..., pattern=r"^D?\d+$")


class FacetInput(Input):
    field: FacetField
    query: str | None = None


# ---------- shared argument types ----------

Rows = Annotated[int, Field(description="Results per page, 1-20. Use 5-10 unless the user asks for more.")]
Offset = Annotated[int, Field(description="Results to skip. Use next_offset from the previous result.")]
NewestFirst = Annotated[bool, Field(description="true = newest documents first; false = most relevant first.")]
OptStr = str | None


# ---------- tools ----------

@mcp.tool
async def search_documents(
    query: Annotated[str, Field(description="Keywords, e.g. 'climate resilience Kenya'.")],
    language: Annotated[OptStr, Field(description="Exact language, e.g. 'English', 'French'.")] = None,
    rows: Rows = 10,
    offset: Offset = 0,
    newest_first: NewestFirst = False,
) -> dict:
    """Keyword search across World Bank document titles, abstracts, countries and project names.

    Use this for open-ended topic questions. Use filter_documents instead when the question names a
    country, region, document type or date range.

    Returns total, next_offset and documents (id, display_title, docdt, docty, majdocty, country, pdfurl, abstract).
    """
    p, err = _validate(SearchInput, query=query, language=language, rows=rows, offset=offset,
                       newest_first=newest_first)
    if err:
        return err
    data = await _call({"qterm": p.query, "lang_exact": p.language, **p.paging()})
    hint = f"No documents matched query={p.query!r}. Try fewer or broader keywords."
    if p.language:
        hint += " Language is case-sensitive; call get_facets(field='language') for exact values."
    return _page(data, p.offset, p.rows, hint)


@mcp.tool
async def filter_documents(
    country: Annotated[OptStr, Field(description="A single country name, e.g. 'Kenya'. For regions like 'Sub-Saharan Africa' or 'East Asia and Pacific', use `region`.")] = None,
    region: Annotated[OptStr, Field(description="Exact region, e.g. 'Sub-Saharan Africa', 'South Asia', 'East Asia and Pacific'.")] = None,
    topic: Annotated[OptStr, Field(description="Exact topic, e.g. 'Education', 'Environment', 'Water Resources'.")] = None,
    document_type: Annotated[OptStr, Field(description="Only set this when the user asks for a specific document type. Leave it empty otherwise.")] = None,
    major_document_type: Annotated[OptStr, Field(description="Exact broad type: 'Project Documents', 'Publications & Research', 'Economic & Sector Work'.")] = None,
    start_date: Annotated[OptStr, Field(description="Earliest date, YYYY-MM-DD.")] = None,
    end_date: Annotated[OptStr, Field(description="Latest date, YYYY-MM-DD.")] = None,
    query: Annotated[OptStr, Field(description="Keywords for the subject, e.g. 'sanitation'. Use this alongside filters when the question names a subject the filters don't cover.")] = None,
    rows: Rows = 10,
    offset: Offset = 0,
    newest_first: NewestFirst = False,
) -> dict:
    """Find World Bank documents by exact country, region, topic, document type and/or date range.

    Use this when the question names any of those. Use search_documents for keyword-only questions,
    and get_facets first if unsure of an exact value (values are case-sensitive).

    Returns the same shape as search_documents.
    """
    p, err = _validate(FilterInput, country=country, region=region, topic=topic, document_type=document_type,
                       major_document_type=major_document_type, start_date=start_date, end_date=end_date,
                       query=query, rows=rows, offset=offset, newest_first=newest_first)
    if err:
        return err
    # Small models swap these two. Regions are a fixed list; countries are checked live
    # (the country facet stops at 100 values, so there is no complete list to hardcode).
    note = None
    if p.country in REGIONS and not p.region:
        note = f"Note: {p.country!r} is a region; searched by region."
        p.region, p.country = p.country, None
    elif p.region and p.region not in REGIONS and not p.country and await _is_country(p.region):
        note = f"Note: {p.region!r} is a country; searched by country."
        p.country, p.region = p.region, None
    filters = {"country": p.country, "region": p.region, "topic": p.topic,
               "document_type": p.document_type, "major_document_type": p.major_document_type}
    api = {FIELDS[k]: v for k, v in filters.items() if v}
    if p.region:
        api[FIELDS["region"]] = REGION_ALIASES.get(p.region, p.region)
    data = await _call({
        **api,
        "qterm": p.query,
        "strdate": p.start_date and p.start_date.isoformat(),
        "enddate": p.end_date and p.end_date.isoformat(),
        **p.paging(),
    })
    used = {k: v for k, v in filters.items() if v}
    shown = ", ".join(f"{k}={v!r}" for k, v in {**used, "query": p.query}.items() if v)
    if used:
        first = next(iter(used))
        hint = (f"No documents matched {shown}. Filter values are case-sensitive; call "
                f"get_facets(field='{first}') to find the exact value, or drop a filter.")
    else:
        hint = f"No documents matched {shown or 'the date range'}. Widen the date range or remove query."
    result = _page(data, p.offset, p.rows, hint)
    if note:
        result = {"note": note, **result}
    return result


@mcp.tool
async def get_document(
    document_id: Annotated[str, Field(description="Document id from a search result, e.g. '33295056'.")],
) -> dict:
    """Get every metadata field for one document: full abstract, authors, keywords, sectors, project, URLs.

    Use this after search_documents or filter_documents when a summary is not enough. Do not use it
    to search; it only accepts an id.
    """
    p, err = _validate(DocumentInput, document_id=document_id)
    if err:
        return err
    doc_id = p.document_id.lstrip("D")
    docs = _documents(await _call({"id": doc_id}))
    if not docs:
        return {"message": f"No document has id {doc_id!r}. Use an id field from search_documents results."}
    doc = {("country" if k == "count" else k): v for k, v in docs[0].items()}  # renamed in place, as in _summary
    if isinstance(doc.get("keywd"), list):  # entries often start with a space; lists run to 100-200
        keywd = [k.strip() if isinstance(k, str) else k for k in doc["keywd"]]
        doc["keywd"] = keywd[:KEYWD_MAX]
        if len(keywd) > KEYWD_MAX:
            doc["keywd_truncated"] = True
    return doc


@mcp.tool
async def get_facets(
    # str + schema enum (not Literal) so a bad value reaches _validate and gets a retry hint.
    field: Annotated[str, Field(description="Field to list values for.", json_schema_extra={"enum": list(FIELDS)})],
    query: Annotated[OptStr, Field(description="Optional keywords; counts only matching documents.")] = None,
) -> dict:
    """List distinct values of a field with document counts, most common first (top 100).

    Use this for "what types/languages/topics exist" or "most frequent" questions, and to get exact
    values for filter_documents. Use search_documents or filter_documents to find actual documents.

    Returns field, total_documents and values [{value, count}].
    """
    p, err = _validate(FacetInput, field=field, query=query)
    if err:
        return err
    api_field = FIELDS[p.field]
    data = await _call({"fct": api_field, "qterm": p.query, "rows": 0})
    raw = data.get("documents", {}).get("facets", {}).get(api_field, {})
    values = [{"value": v["name"], "count": v["count"]} for v in raw.values()]
    result = {"field": p.field, "total_documents": int(data.get("total", 0)), "values": values}
    if not values:
        result["message"] = f"No values for field={p.field!r} with query={p.query!r}. Remove or broaden query."
    return result


if __name__ == "__main__":
    mcp.run(show_banner=False)  # keep the client's stderr trace clean
