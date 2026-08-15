import textwrap

import numpy as np

from retrieval_databases import hybrid_retriever
from reranking import rerank_chunks
from mmr import maximal_marginal_relevance


REPORT_WIDTH = 100

# A resume is a small document - the one used for tuning holds 26 chunks. Below
# this size the cross-encoder reads every chunk instead of only the ones
# retrieval shortlisted, which costs about 0.45 s per criterion on CPU and
# removes the recall bottleneck completely. Measured on the computer-vision
# requirement: shortlisting 12 chunks first put ZERO of the three real CV
# projects in front of the judge, because the shortlist depended on one loosely
# worded search query. Reranking all 26 recovered the CNN classifier.
RERANK_ALL_LIMIT = 60


def write_block(f, text, prefix):
    """Write a possibly multi-line block with EVERY line indented.

    Chunk text carries its own newlines for bullet lists. Writing it raw
    indented only the first line and left the rest at column zero, which is
    what made the old report unreadable.
    """
    for line in text.split("\n"):

        line = line.rstrip()

        if line == "":
            f.write("\n")
            continue

        # A wrapped bullet should hang under its own text; a wrapped ordinary
        # sentence should not be indented any further than the line it continues.
        if line.lstrip().startswith("- "):
            continuation = prefix + "  "
        else:
            continuation = prefix

        wrapped_lines = textwrap.wrap(
            line,
            width=REPORT_WIDTH,
            initial_indent=prefix,
            subsequent_indent=continuation
        )

        for wrapped in wrapped_lines:
            f.write(wrapped + "\n")


def write_field(f, label, value):
    """One aligned "LABEL : value" line, wrapped under the value column."""
    prefix = f"  {label:<12}: "
    padding = " " * len(prefix)

    wrapped_lines = textwrap.wrap(
        str(value),
        width=REPORT_WIDTH,
        initial_indent=prefix,
        subsequent_indent=padding
    )

    for wrapped in wrapped_lines:
        f.write(wrapped + "\n")


def write_report_header(f, groups, chunks, resume_file, candidate_name, top_k, final_k):

    criteria_count = sum(len(group["criteria"]) for group in groups)

    f.write("=" * REPORT_WIDTH + "\n")
    f.write("  RESUME SCREENING - RETRIEVAL REPORT\n")
    f.write("=" * REPORT_WIDTH + "\n")

    write_field(f, "Resume", resume_file)
    write_field(f, "Candidate", candidate_name)
    write_field(f, "Chunks", len(chunks))
    write_field(f, "Groups", f"{len(groups)} (one search each)")
    write_field(f, "Criteria", f"{criteria_count} (one verdict each)")
    write_field(f, "Retrieved", f"{top_k} per group")
    write_field(f, "Kept", f"{final_k} per group after reranking + MMR")

    f.write("=" * REPORT_WIDTH + "\n\n")

    # Index first, so the whole shortlist is scannable without scrolling
    # through every evidence block.
    f.write("  SUMMARY OF REQUIREMENT GROUPS\n")
    f.write("  " + "-" * (REPORT_WIDTH - 2) + "\n")
    f.write(f"  {'#':>3}  {'PRIORITY':<9} {'RETRIEVAL':<10} {'CRITERIA':<9} GROUP\n")
    f.write("  " + "-" * (REPORT_WIDTH - 2) + "\n")

    for number, group in enumerate(groups, start=1):

        f.write(
            f"  {number:>3}  {group['priority']:<9} {group['retrieval_type']:<10}"
            f" {len(group['criteria']):<9} {group['name']}\n"
        )

    f.write("  " + "-" * (REPORT_WIDTH - 2) + "\n\n\n")


def build_evidence(
    groups,
    chunks,
    dense_vector_db,
    sparse_db,
    tfidf,
    resume_file="",
    candidate_name="unknown",
    top_k=8,
    final_k=4,
    lambda_mult=0.8,
    output_path="output_file.txt"
):
    """Retrieve and rerank once per requirement group.

    Retrieval casts a wide net with `top_k` candidates. The cross-encoder then
    reads each candidate together with the group's search text and reorders
    them, which is far more accurate than the bi-encoder but too slow to run
    over every chunk - so it only ever sees the shortlist.

    One search per group is the whole point of grouping: requirements that the
    same resume lines would answer share one search and one grading call, while
    still being graded one by one.

    Reranking sorts by relevance alone, so the top 4 can easily be four chunks
    saying the same thing. MMR picks the final set by trading relevance against
    how similar a candidate already is to what has been picked, which is what
    gets a project AND an experience entry in front of the judge instead of two
    near-identical skills lists.

    Writes the readable report to `output_path` and returns a NEW list of
    groups, each with a "chunks" key holding the chunk dicts that survived.
    Chunks stay as full dicts so the judge sees chunk_id and section too.
    """
    # Copy every group instead of writing into the caller's list. The same
    # groups object is reused for all 25 resumes, so mutating it would leave
    # resume 1's chunks sitting inside the groups handed to resume 2.
    groups = [dict(group) for group in groups]

    with open(output_path, "w", encoding="utf8") as f:

        write_report_header(
            f, groups, chunks, resume_file, candidate_name, top_k, final_k
        )

        for number, group in enumerate(groups, start=1):

            f.write("#" * REPORT_WIDTH + "\n")
            f.write(f"  GROUP {number} of {len(groups)}: {group['name']}\n")
            f.write("#" * REPORT_WIDTH + "\n\n")

            write_field(f, "Priority", group["priority"])
            write_field(f, "Retrieval", group["retrieval_type"])
            f.write("\n")

            f.write("  CRITERIA GRADED FROM THIS GROUP'S EVIDENCE\n")
            f.write("  " + "-" * (REPORT_WIDTH - 2) + "\n")

            for criterion_number, criterion in enumerate(group["criteria"], start=1):
                write_block(f, f"{criterion_number}. {criterion}", "    ")

            f.write("\n")

            f.write("  SEARCH QUERIES\n")
            f.write("  " + "-" * (REPORT_WIDTH - 2) + "\n")

            for query_number, query in enumerate(group["retrieval_queries"], start=1):
                write_block(f, f"{query_number}. {query}", "    ")

            if len(group["retrieval_queries"]) == 0:
                write_block(
                    f,
                    "(none generated - only the group name was searched)",
                    "    "
                )

            f.write("\n")

            # The group name is the search text the cross-encoder scores
            # against. A single criterion cannot do that job, because a group
            # holds several and picking one would bias the evidence towards
            # whichever happened to come first.
            top_indices, final_similarity, dense_raw_scores, sparse_raw_scores = hybrid_retriever(
                group["name"],
                group["retrieval_queries"],
                group["retrieval_type"],
                top_k,
                dense_vector_db,
                sparse_db,
                tfidf
            )

            if len(top_indices) == 0:

                group["chunks"] = ()

                f.write("  NO EVIDENCE RETRIEVED\n")
                f.write("  " + "-" * (REPORT_WIDTH - 2) + "\n\n\n")
                continue

            # Rerank against the CRITERIA, not the group name. The group name is
            # whatever the decomposer called it, and a vague one like "Hands-on
            # project experience" gives the cross-encoder nothing to work with:
            # measured on this resume it scored every chunk within 1.9 of every
            # other and ranked a Bluetooth car above a CNN classifier for the
            # computer-vision requirement. Handing it the criterion text instead
            # widened the spread to 5.2 and moved the CNN project from 5th to
            # 2nd. Same model, same chunks - only the query changed.
            rerank_query = " ".join(group["criteria"])

            # On a small document the shortlist is skipped entirely and the
            # cross-encoder judges every chunk. Retrieval still runs, because
            # its scores are what the report shows and what a larger document
            # would fall back on.
            if len(chunks) <= RERANK_ALL_LIMIT:
                rerank_pool = np.arange(len(chunks))
            else:
                rerank_pool = top_indices

            reranked_indices, reranked_scores = rerank_chunks(
                rerank_query,
                rerank_pool,
                chunks
            )

            mmr_indices, mmr_scores, normalized_scores = maximal_marginal_relevance(
                reranked_indices,
                reranked_scores,
                dense_vector_db,
                final_k=final_k,
                lambda_mult=lambda_mult
            )

            group["chunks"] = tuple(
                chunks[index]
                for index in mmr_indices
            )

            kept = set(int(index) for index in mmr_indices)

            f.write(
                f"  EVIDENCE AFTER RERANKING"
                f"   ({len(reranked_indices)} of {len(chunks)} chunks scored,"
                f" top {len(group['chunks'])} kept by MMR)\n"
            )
            f.write("  " + "-" * (REPORT_WIDTH - 2) + "\n")

            for rank, index in enumerate(reranked_indices, start=1):

                chunk = chunks[index]

                # MMR can keep a lower-ranked chunk and drop a higher one,
                # so membership is checked rather than assumed from position.
                if int(index) in kept:
                    marker = "SELECTED"
                else:
                    marker = "dropped"

                f.write("\n")
                f.write(
                    f"    RANK {rank}  [{marker}]"
                    f"   {chunk['chunk_id']}   [section: {chunk['section']}]\n"
                )
                f.write(
                    f"    scores    rerank {reranked_scores[rank - 1]:>8.4f}"
                    f"   hybrid {final_similarity[index]:.5f}"
                    f"   dense {dense_raw_scores[index]:.3f}"
                    f"   sparse {sparse_raw_scores[index]:.3f}\n"
                )
                f.write("    " + "." * (REPORT_WIDTH - 4) + "\n")

                write_block(f, chunk["text"], "      ")

            f.write("\n\n")

    return groups
