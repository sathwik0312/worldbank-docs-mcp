"""Replay checker for eval runs: is every answer grounded in what the model actually saw?

    python scripts/check_citations.py              # grade eval/runs/q*_r*.txt
    python scripts/check_citations.py --self-test  # plant fakes, confirm they are caught

For each run it re-executes the logged tool calls against the MCP server, applies the client's
own fit() truncation (so it sees exactly what the model saw), then checks the answer:
- every cited document title exists in those results, with the date the answer gives;
- every link in the answer is a pdfurl from those results;
- per-question criteria (fixed before the runs were graded), e.g. Q3 top 5 in order with correct
  counts if given, Q5 lists >=95% of returned languages. Counts are optional; any count given must be right.

Caveat: the API breaks relevance ties non-deterministically, so a replay can return a different
document than the run saw (see eval/results.md). Re-run the checker on a single failure before
calling it a hallucination.
"""

import argparse
import asyncio
import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "client"))
import agent  # noqa: E402  (reuse connect() and fit() so the replay matches the client exactly)
from run_eval import ANSWER_HEADER, DEFAULT_OUT, QUESTIONS, TRACE_HEADER  # noqa: E402

URL = re.compile(r"https?://[^\s)\]>]+")
LIST_ITEM = r"[ ]?(?:\d+\.|[-*])\s"  # top-level only: sub-bullets are indented
SSA = {"Sub-Saharan Africa", "Eastern and Southern Africa", "Western and Central Africa", "Africa"}
DOC_QUESTIONS = (1, 2, 4, 6)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def strip_url(u: str) -> str:
    return re.sub(r"^https?://", "", u)


def to_iso(s: str):
    s = s.replace("*", "").strip().rstrip(".")[:30].strip()
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%d %B %Y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            pass
    m = re.match(r"(\d{4}-\d{2}-\d{2})", s)
    return m.group(1) if m else None


def items(answer: str) -> list[tuple[str, str | None, str]]:
    """Top-level list entries (sub-bullets are indented): (title, iso date or None, block text)."""
    out = []
    for block in re.split(rf"\n(?={LIST_ITEM})", "\n" + answer):
        block = block.strip("\n")
        if not re.match(LIST_ITEM, block):
            continue
        first = block.splitlines()[0]
        bold = [x for x in re.findall(r"\*\*(.+?)\*\*", first) if not re.fullmatch(r"\s*title:?\s*", x, re.I)]
        title = bold[0] if bold else re.sub(r"^\s*(?:\d+\.|[-*])\s+", "", first)
        title = re.sub(r"^title:?\s*", "", title.strip(" *:"), flags=re.I).strip(' *:"')
        m = re.search(r"(?:date|published)[^:\n]*:\**\s*([^\n|]+)", block, re.I)
        out.append((title, to_iso(m.group(1)) if m else None, block))
    return out


def name_and_counts(title: str, block: str) -> tuple[str, list[int]]:
    """'English (578,336 documents)' or '**Brief**: 24,818' -> ('English', [578336])."""
    name = re.sub(r"\s*[:(\-–]?\s*\d[\d,]*\s*(?:documents?)?\)?\s*$", "", title).strip(" *:-")
    body = re.sub(rf"^{LIST_ITEM}+", "", block)  # the list number is not a count
    return name, [int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", body) if x.replace(",", "").isdigit()]


async def call_json(session, name: str, args: dict) -> dict:
    return json.loads(agent.result_text(await session.call_tool(name, args)))


async def replay(session, text: str):
    """Re-run each logged tool call; return calls plus the documents/facet values the model saw."""
    calls, docs, values, notes = [], [], [], []
    for name, args in re.findall(r"^-> (\w+) (\{.*\})$", text, re.M):
        args = json.loads(args)
        calls.append((name, args))
        seen = agent.fit(agent.result_text(await session.call_tool(name, args)))
        try:
            data = json.loads(seen)
        except ValueError:
            continue
        docs += data.get("documents", [])
        values += data.get("values", [])
        if data.get("note"):
            notes.append(data["note"])
    return calls, docs, values, notes


def check_citations(answer: str, docs: list[dict]):
    """Return (cited docs, problems, links not in results)."""
    visible = [(norm(d.get("display_title")), d) for d in docs]
    visible_urls = {strip_url(d.get("pdfurl", "")) for d in docs}
    fake_urls = [u for u in URL.findall(answer) if strip_url(u) not in visible_urls]
    cited, problems = [], []
    for title, date, _ in items(answer):
        t = norm(title)[:45]
        if len(t) < 12:  # not a document entry
            continue
        hits = [d for vt, d in visible if vt.startswith(t) or (len(vt) > 12 and t.startswith(vt[:45]))]
        if not hits:
            problems.append(f"title not in results: {title[:60]!r}")
            continue
        exact = [d for d in hits if date is None or d["docdt"][:10] == date]
        if not exact:
            problems.append(f"date {date} wrong for {title[:50]!r} "
                            f"(results: {sorted({d['docdt'][:10] for d in hits})[:3]})")
            continue
        cited.append(exact[0])
    return cited, problems, fake_urls


def count_ok(given: list[int], actual: int) -> bool:
    """Live counts grow as documents are added (e.g. +2 between a run and its replay the same day),
    so a count matches within 0.1% (min 5). Invented or misread counts are far outside that."""
    return any(abs(g - actual) <= max(5, actual // 1000) for g in given)


def check_facet_answer(answer: str, values: list[dict], top_n: int | None, min_share: float):
    """Facet answers: listed names must be real values, any counts given must match; optionally top-N order."""
    actual = {norm(v["value"]): v["count"] for v in values}
    order = [norm(v["value"]) for v in values]
    listed = [(norm(n), c) for n, c in (name_and_counts(t, b) for t, _, b in items(answer)) if norm(n)]
    fails = []
    if top_n and [n for n, _ in listed[:top_n]] != order[:top_n]:
        fails.append(f"top {top_n} wrong: {[n for n, _ in listed[:top_n]]}")
    invented = [n for n, _ in listed if n not in actual]
    if invented:
        fails.append(f"values not in results: {invented[:5]}")
    # Counts are optional; any count the answer does give must be right.
    no_count = [n for n, c in listed if n in actual and not c]
    wrong = [f"{n}: {c} vs {actual[n]}" for n, c in listed if n in actual and c and not count_ok(c, actual[n])]
    if wrong:
        fails.append(f"wrong counts: {wrong[:5]}")
    covered = len({n for n, _ in listed} & set(actual))
    if covered < min_share * len(actual):
        fails.append(f"listed {covered} of {len(actual)} values")
    return fails, {"listed": f"{covered}/{len(actual)}", "with counts": len(listed) - len(no_count)}


async def newest_esw_indonesia(session) -> tuple[set[str], str]:
    docs = (await call_json(session, "filter_documents", {"country": "Indonesia", "newest_first": True, "rows": 10,
                                                          "major_document_type": "Economic & Sector Work"}))["documents"]
    top = docs[0]["docdt"][:10]
    return {norm(d["display_title"]) for d in docs if d["docdt"][:10] == top}, top


async def grade(session, q: int, text: str, newest: tuple[set[str], str]) -> dict:
    answer = text.split(ANSWER_HEADER, 1)[1].strip()
    trace = text.split(TRACE_HEADER, 1)[1].split(ANSWER_HEADER, 1)[0]
    calls, docs, values, notes = await replay(session, text)
    cited, problems, fake_urls = check_citations(answer, docs) if q in DOC_QUESTIONS else ([], [], [])
    fails, extra = problems + [f"link not in results: {u[:70]}" for u in fake_urls], {}

    if not cited and not values and re.search(
            r"would you like|could you (please )?(clarify|specify|provide)|do you want me to", answer, re.I):
        fails.append("asked a clarifying question instead of answering")
    if q in (1, 2, 6) and len(cited) < 3:
        fails.append(f"only {len(cited)} real documents cited")
    if q == 1:
        if not any(re.search(r"climate|resilien", c["display_title"], re.I) for c in cited):
            fails.append("no climate/resilience document cited")
        if any("Kenya" not in str(c.get("country")) and "kenya" not in norm(c["display_title"]) for c in cited):
            fails.append("non-Kenya document cited")
    if q == 2:
        outside = [c["docdt"][:10] for c in cited if not "2019-01-01" <= c["docdt"][:10] <= "2022-12-31"]
        if outside:
            fails.append(f"cited docs outside 2019-2022: {outside}")
        if not any(a.get("region") in SSA or a.get("country") in SSA or "Sub-Saharan" in json.dumps(a) for _, a in calls):
            fails.append("never searched Sub-Saharan Africa")
        edu = sum(bool(re.search(r"educat|school|learn|skill|teach", c["display_title"], re.I)) for c in cited)
        extra["education titles"] = f"{edu}/{len(cited)}"
    if q == 3:
        f, extra = check_facet_answer(answer, values, top_n=5, min_share=0)
        fails += f
    if q == 4:
        titles, date = newest
        if not any(t[:40] in norm(answer) for t in titles):
            fails.append(f"did not cite a newest ({date}) ESW document")
    if q == 5:
        f, extra = check_facet_answer(answer, values, top_n=None, min_share=0.95)
        fails += f
    if q == 6:
        if any("Brazil" not in str(c.get("country")) for c in cited):
            fails.append("non-Brazil document cited")
        water = sum(bool(re.search(r"water|sanit|sewer|agua", c["display_title"], re.I)) for c in cited)
        if cited and water < len(cited) / 2:
            fails.append(f"only {water}/{len(cited)} cited titles about water/sanitation")
        extra["water/sanitation titles"] = f"{water}/{len(cited)}"
    return {
        "calls": [f"{n}({', '.join(f'{k}={v!r}' for k, v in a.items() if v not in (None, False))})" for n, a in calls],
        "fails": fails, "cited": len(cited), "notes": notes, "extra": extra,
        "links": len(URL.findall(answer)), "trimmed": trace.count("trimmed to"), "capped": "hit 8 iterations" in trace,
    }


def fake_run(call: tuple[str, dict], answer: str) -> str:
    name, args = call
    return f"QUESTION: self-test\n\n{TRACE_HEADER}\n-> {name} {json.dumps(args)}\n\n{ANSWER_HEADER}\n{answer}"


async def self_test(session) -> bool:
    """Plant known errors in answers built from live results and confirm each one is caught."""
    newest = await newest_esw_indonesia(session)
    ok = True

    def expect(label, condition):
        nonlocal ok
        ok &= condition
        print(f"  {'ok  ' if condition else 'FAIL'} {label}")

    call = ("filter_documents", {"country": "Brazil", "query": "sanitation", "rows": 5})
    docs = (await call_json(session, *call))["documents"]
    a, b = docs[0], docs[1]
    answer = (f"1. **{a['display_title']}**\n   - Date: {a['docdt'][:10]}\n   - [PDF]({a['pdfurl']})\n"
              f"2. **Brazil - Imaginary Sanitation Megaproject for Testing**\n   - Date: 2020-01-01\n"
              f"3. **{b['display_title']}**\n   - Date: 1901-01-01\n"
              f"   - [PDF](http://documents.worldbank.org/curated/en/000/pdf/fake.pdf)\n")
    r = await grade(session, 6, fake_run(call, answer), newest)
    print("documents (1 real entry, 3 planted errors):")
    expect("real entry accepted", r["cited"] == 1)
    expect("invented title caught", any("Imaginary" in f for f in r["fails"]))
    expect("wrong date caught", any("1901-01-01" in f for f in r["fails"]))
    expect("fake link caught", any("fake.pdf" in f for f in r["fails"]))

    call = ("get_facets", {"field": "document_type"})
    values = (await call_json(session, *call))["values"]
    good = "\n".join(f"{i}. **{v['value']}**: {v['count']:,}" for i, v in enumerate(values[:5], 1))
    swapped = [values[1], values[0]] + values[2:5]
    bad = "\n".join(f"{i}. **{v['value']}**: {v['count'] + (5000 if i == 3 else 0):,}" for i, v in enumerate(swapped, 1))
    print("facets (correct answer, then swapped order + wrong count):")
    expect("correct top 5 with counts accepted", not (await grade(session, 3, fake_run(call, good), newest))["fails"])
    bad_fails = (await grade(session, 3, fake_run(call, bad), newest))["fails"]
    expect("swapped order caught", any("top 5 wrong" in f for f in bad_fails))
    expect("wrong count caught", any("wrong counts" in f for f in bad_fails))
    return ok


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs_dir", nargs="?", type=Path, default=DEFAULT_OUT, help="default eval/runs")
    ap.add_argument("--self-test", action="store_true", help="check that planted errors are caught")
    args = ap.parse_args()

    async with agent.connect() as session:
        if args.self_test:
            ok = await self_test(session)
            print("self-test", "passed" if ok else "FAILED")
            return 0 if ok else 1

        files = sorted(args.runs_dir.glob("q[0-9]_r[0-9]*.txt"))
        if not files:
            print(f"no runs in {args.runs_dir}; run scripts/run_eval.py first")
            return 1
        newest = await newest_esw_indonesia(session)
        passes, totals = Counter(), Counter()
        for f in files:
            q = int(f.name[1])
            r = await grade(session, q, f.read_text(), newest)
            totals[q] += 1
            passes[q] += not r["fails"]
            print(f"{f.stem} {'PASS' if not r['fails'] else 'FAIL'} cited={r['cited']} links={r['links']} "
                  f"trimmed={r['trimmed']} capped={r['capped']} {r['extra']}")
            for line in r["calls"] + [f"note: {n}" for n in r["notes"]] + [f"FAIL: {x}" for x in r["fails"]]:
                print(f"      {line[:170]}")
        print("\nsummary")
        for q in sorted(totals):
            print(f"  Q{q} {passes[q]}/{totals[q]}  {QUESTIONS[q - 1]}")
        return 0 if passes == totals else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
