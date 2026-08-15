from datetime import datetime

from build_evidence import write_block, write_field, REPORT_WIDTH


def write_verdict(evidence, resume_file, candidate_name, final_score, status,
                  output_path="verdict.txt", top_n=2):
    """One short per-candidate file: name, score, and the best evidence found.

    The evidence report is the full audit trail and the gradesheet is the
    per-criterion breakdown. This is the third file - the one a human actually
    reads first. It keeps only the `top_n` chunks each search kept, in the order
    MMR ranked them.
    """
    with open(output_path, "w", encoding="utf8") as f:

        f.write("=" * REPORT_WIDTH + "\n")
        f.write("  CANDIDATE VERDICT\n")
        f.write("=" * REPORT_WIDTH + "\n")

        write_field(f, "Candidate", candidate_name)
        write_field(f, "Resume", resume_file)
        write_field(f, "Score", f"{final_score:.2f} / 100")
        write_field(f, "Status", status)

        f.write("=" * REPORT_WIDTH + "\n\n")

        f.write(f"  TOP {top_n} RETRIEVED CHUNKS PER SEARCH\n")
        f.write("  " + "-" * (REPORT_WIDTH - 2) + "\n\n")

        for number, group in enumerate(evidence, start=1):

            f.write(f"  GROUP {number}: {group['name']}  [{group['priority']}]\n")
            f.write("  " + "-" * (REPORT_WIDTH - 2) + "\n")

            for query_number, query in enumerate(group["retrieval_queries"], start=1):
                write_block(f, f"query {query_number}: {query}", "    ")

            if len(group["retrieval_queries"]) == 0:
                write_block(f, "query 1: (group name only)", "    ")

            f.write("\n")

            chunks = group.get("chunks", ())

            if len(chunks) == 0:
                write_block(f, "no evidence retrieved", "    ")
                f.write("\n\n")
                continue

            for rank, chunk in enumerate(chunks[:top_n], start=1):

                f.write(
                    f"    [{rank}] {chunk['chunk_id']}"
                    f"   [section: {chunk['section']}]\n"
                )

                write_block(f, chunk["text"], "        ")
                f.write("\n")

            f.write("\n")


def write_ranking(results, job_description_file, shortlist_size,
                  output_path="pipeline_ranking.txt"):
    """Every resume in one ranked table, and the sorted list back to the caller.

    Resumes that could not be read have score None. They are kept out of the
    ranking - an unreadable scan is not a zero - but listed underneath so a
    human still sees them.
    """
    scored = [result for result in results if result["score"] is not None]
    unscored = [result for result in results if result["score"] is None]

    scored.sort(key=lambda result: result["score"], reverse=True)

    with open(output_path, "w", encoding="utf8") as f:

        f.write("=" * REPORT_WIDTH + "\n")
        f.write("  PIPELINE RANKING\n")
        f.write("=" * REPORT_WIDTH + "\n")

        write_field(f, "Job", job_description_file)
        write_field(f, "Screened", f"{len(results)} resumes")
        write_field(f, "Ranked", f"{len(scored)} scored, {len(unscored)} could not be read")
        write_field(f, "Generated", datetime.now().strftime("%d %b %Y, %I:%M %p"))

        f.write("=" * REPORT_WIDTH + "\n\n")

        f.write(f"  {'#':>3}  {'SCORE':>6}  {'CANDIDATE':<24} FILE\n")
        f.write("  " + "-" * (REPORT_WIDTH - 2) + "\n")

        for rank, result in enumerate(scored, start=1):

            f.write(
                f"  {rank:>3}  {result['score']:>6.2f}  {result['name']:<24}"
                f" {result['file']}\n"
            )

            if result["status"] != "ok":
                write_block(f, result["status"], "        ")

        f.write("  " + "-" * (REPORT_WIDTH - 2) + "\n\n")

        f.write(f"  SHORTLISTED FOR INTERVIEW (top {shortlist_size})\n")
        f.write("  " + "-" * (REPORT_WIDTH - 2) + "\n")

        for rank, result in enumerate(scored[:shortlist_size], start=1):
            f.write(f"  {rank}. {result['name']:<24} {result['score']:>6.2f}   {result['file']}\n")

        f.write("\n")

        if len(unscored) > 0:

            f.write("  NOT RANKED - NEEDS A HUMAN\n")
            f.write("  " + "-" * (REPORT_WIDTH - 2) + "\n")

            for result in unscored:
                f.write(f"  {result['file']}\n")
                write_block(f, result["status"], "      ")

            f.write("\n")

    return scored
