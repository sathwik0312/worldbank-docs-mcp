"""Run the TASK.md test questions through client/agent.py N times each and save every run.

    python scripts/run_eval.py                  # all 6 questions x 3 runs -> eval/runs/
    python scripts/run_eval.py --runs 1 --questions 3 5

Each run is saved as eval/runs/q{question}_r{run}.txt: the question, the client's stderr trace
(tool calls and result sizes), and the final answer. Check them with scripts/check_citations.py.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLIENT = ROOT / "client" / "agent.py"
DEFAULT_OUT = ROOT / "eval" / "runs"
TRACE_HEADER = "=== TRACE (stderr) ==="
ANSWER_HEADER = "=== ANSWER (stdout) ==="

QUESTIONS = [
    "What World Bank reports exist on climate resilience in Kenya?",
    "Find documents about education financing in Sub-Saharan Africa published between 2019 and 2022.",
    "What document types does the World Bank publish most frequently?",
    "Find the most recent economic sector work on Indonesia.",
    "List all languages the World Bank publishes documents in.",
    "Find project documents related to water and sanitation in Brazil.",
]


def run_once(question: str, timeout: int) -> tuple[str, str]:
    try:
        p = subprocess.run([sys.executable, str(CLIENT), question], capture_output=True, text=True,
                           timeout=timeout, cwd=ROOT)
        return p.stderr, p.stdout
    except subprocess.TimeoutExpired as e:
        return (e.stderr or "") + f"\n(timed out after {timeout}s)", ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=3, help="runs per question (default 3)")
    ap.add_argument("--questions", type=int, nargs="+", default=range(1, len(QUESTIONS) + 1),
                    choices=range(1, len(QUESTIONS) + 1), metavar="N", help="question numbers 1-6 (default all)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory (default eval/runs)")
    ap.add_argument("--timeout", type=int, default=600, help="seconds per run (default 600)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for q in args.questions:
        for r in range(1, args.runs + 1):
            start = time.time()
            trace, answer = run_once(QUESTIONS[q - 1], args.timeout)
            path = args.out / f"q{q}_r{r}.txt"
            path.write_text(f"QUESTION: {QUESTIONS[q - 1]}\n\n{TRACE_HEADER}\n{trace}\n{ANSWER_HEADER}\n{answer}")
            print(f"q{q} r{r}: {time.time() - start:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
