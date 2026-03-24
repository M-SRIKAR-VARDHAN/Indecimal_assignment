"""Evaluation script: runs 15 test questions through the RAG pipeline and scores quality."""

import os
import sys
import time

# Add parent dir to path so we can import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from rag_engine import RAGEngine

# ── Test questions with expected key terms ────────────────────────────────────

TEST_CASES = [
    {
        "question": "What packages does Indecimal offer and what are their per-sqft prices?",
        "expected_terms": ["essential", "premier", "infinia", "pinnacle", "1,851", "1,995", "2,250", "2,450"],
    },
    {
        "question": "What brand of steel is used in the Essential package?",
        "expected_terms": ["sunvik", "kamadhenu", "68,000"],
    },
    {
        "question": "How does Indecimal's escrow payment model work?",
        "expected_terms": ["escrow", "project manager", "verif", "disburse"],
    },
    {
        "question": "What is the ceiling height across all packages?",
        "expected_terms": ["10 ft", "floor-to-floor"],
    },
    {
        "question": "What paint brand is used for interior painting in the Pinnacle package?",
        "expected_terms": ["royale", "asian paints"],
    },
    {
        "question": "How does Indecimal handle construction delays?",
        "expected_terms": ["zero-tolerance", "tracking", "penalis", "flag"],
    },
    {
        "question": "What are the bathroom sanitary fitting allowances for each package?",
        "expected_terms": ["32,000", "50,000", "70,000", "80,000"],
    },
    {
        "question": "What is included in the zero cost maintenance program?",
        "expected_terms": ["plumbing", "electrical", "painting", "roofing"],
    },
    {
        "question": "What type of windows are offered in the Premier package?",
        "expected_terms": ["upvc", "500", "lesso"],
    },
    {
        "question": "How many quality checkpoints does Indecimal have?",
        "expected_terms": ["445"],
    },
    {
        "question": "What is the main door specification for the Infinia package?",
        "expected_terms": ["teak", "40,000"],
    },
    {
        "question": "How does home financing work with Indecimal?",
        "expected_terms": ["relationship manager", "documentation", "7 days", "30 days"],
    },
    {
        "question": "What flooring options are available for living and dining areas?",
        "expected_terms": ["tiles", "granite", "marble"],
    },
    {
        "question": "What is the partner onboarding process?",
        "expected_terms": ["verification", "background", "agreement", "onboarding"],
    },
    {
        "question": "What kitchen sink faucet brands are used in the Essential vs Pinnacle packages?",
        "expected_terms": ["parryware", "hindware", "jaquar", "essco"],
    },
]


def evaluate_retrieval(chunks: list[dict], expected_terms: list[str]) -> bool:
    """Check if retrieved chunks contain expected key terms."""
    combined = " ".join(c["text"] for c in chunks).lower()
    hits = sum(1 for t in expected_terms if t.lower() in combined)
    return hits >= max(1, len(expected_terms) // 2)


def evaluate_hallucination(answer: str, chunks: list[dict]) -> bool:
    """Basic check: does the answer contain claims not found in chunks?
    Returns True if there is hallucination RISK (bad)."""
    combined_context = " ".join(c["text"] for c in chunks).lower()
    # Extract numbers from answer that should be grounded
    import re
    numbers_in_answer = re.findall(r"₹[\d,]+", answer)
    for num in numbers_in_answer:
        if num not in combined_context and num.replace("₹", "") not in combined_context:
            return True  # hallucination risk
    return False


def evaluate_completeness(answer: str) -> bool:
    """Non-empty, reasonable length."""
    return len(answer.strip()) > 30 and not answer.startswith("Error")


def evaluate_groundedness(answer: str) -> bool:
    """Does the answer cite sources?"""
    lower = answer.lower()
    return any(ref in lower for ref in ["doc1", "doc2", "doc3", "source", "document", "according"])


def main():
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key or api_key == "your_key_here":
        print("WARNING: No OPENROUTER_API_KEY found in .env -- set it before running eval.")
        sys.exit(1)

    print("Initializing RAG engine...")
    engine = RAGEngine(api_key=api_key)
    engine.initialize()
    print(f"Ready — {len(engine.chunks)} chunks indexed.\n")

    results = []
    total_time = 0

    for i, tc in enumerate(TEST_CASES, 1):
        q = tc["question"]
        print(f"[{i:2d}/15] {q}")

        result = engine.query(q)
        answer = result["answer"]
        chunks = result["retrieved_chunks"]
        resp_time = result.get("response_time", 0)
        total_time += resp_time

        retrieval_ok = evaluate_retrieval(chunks, tc["expected_terms"])
        hallucination_risk = evaluate_hallucination(answer, chunks)
        complete = evaluate_completeness(answer)
        grounded = evaluate_groundedness(answer)

        results.append(
            {
                "num": i,
                "question": q,
                "retrieval_ok": retrieval_ok,
                "hallucination_risk": hallucination_risk,
                "complete": complete,
                "grounded": grounded,
                "resp_time": resp_time,
                "answer_preview": answer[:120].replace("\n", " "),
            }
        )
        status = "PASS" if (retrieval_ok and not hallucination_risk and complete) else "WARN"
        print(f"       [{status}] retrieval={'OK' if retrieval_ok else 'MISS'} | "
              f"halluc={'RISK' if hallucination_risk else 'OK'} | "
              f"complete={'YES' if complete else 'NO'} | "
              f"grounded={'YES' if grounded else 'NO'} | {resp_time}s\n")
        # Small delay to respect rate limits
        time.sleep(1)

    # ── Summary ──────────────────────────────────────────────────────────────
    ret_ok = sum(1 for r in results if r["retrieval_ok"])
    no_halluc = sum(1 for r in results if not r["hallucination_risk"])
    comp_ok = sum(1 for r in results if r["complete"])
    ground_ok = sum(1 for r in results if r["grounded"])
    avg_time = total_time / len(results) if results else 0

    print("=" * 70)
    print(f"RESULTS SUMMARY")
    print(f"  Retrieval hits:      {ret_ok}/15")
    print(f"  No hallucination:    {no_halluc}/15")
    print(f"  Complete answers:    {comp_ok}/15")
    print(f"  Grounded answers:    {ground_ok}/15")
    print(f"  Avg response time:   {avg_time:.2f}s")
    print("=" * 70)

    # ── Save to markdown ─────────────────────────────────────────────────────
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(eval_dir, "eval_results.md")

    lines = [
        "# RAG Evaluation Results\n",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n",
        f"**Chunks indexed:** {len(engine.chunks)}\n\n",
        "| # | Question | Retrieval OK | Hallucination Risk | Complete | Grounded | Time (s) |",
        "|---|----------|:---:|:---:|:---:|:---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['num']} | {r['question'][:60]}… | "
            f"{'✅' if r['retrieval_ok'] else '❌'} | "
            f"{'⚠️' if r['hallucination_risk'] else '✅'} | "
            f"{'✅' if r['complete'] else '❌'} | "
            f"{'✅' if r['grounded'] else '❌'} | "
            f"{r['resp_time']:.1f} |"
        )
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Retrieval hits: **{ret_ok}/15**")
    lines.append(f"- No hallucination risk: **{no_halluc}/15**")
    lines.append(f"- Complete answers: **{comp_ok}/15**")
    lines.append(f"- Grounded answers: **{ground_ok}/15**")
    lines.append(f"- Average response time: **{avg_time:.2f}s**")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
