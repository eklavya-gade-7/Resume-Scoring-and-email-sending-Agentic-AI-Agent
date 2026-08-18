from get_data_from_pdf import get_profile, ExtractionFailed
from chunk_resume import create_structural_chunks_from_resume
from create_chunk_embeddings import create_embeddings
from decompose_JD import query_decomposition
from retrieval_databases import create_sparse_db
from build_evidence import build_evidence
from generate_score import generate_score, provider_summary
from write_verdict import write_verdict, write_ranking
from bias_check import run_bias_check
from agent import start_candidate_workflows
from google_calender import proposed_slots
import os
import sys
import json
import hashlib
import subprocess
import traceback


cwd = os.getcwd()

JOB_DESCRIPTION_FILE = "job_description_ai_ml.txt"

with open(os.path.join(cwd, JOB_DESCRIPTION_FILE), encoding="utf8") as f:
    job_description = f.read()

# 12, not 8. The reranker is local and fast, so widening what it gets to look
# at costs nothing but gives it a real chance of finding the right chunk. On a
# 26-chunk resume, 8 was only the top 30% of the document.
TOP_K = 12

# 5, not 4. With the reranker now given the criterion text it puts the right
# chunk in the top 4 - but only just, at rank 4 on the deep-learning-framework
# group. One extra slot is cheap insurance against MMR dropping it.
FINAL_K = 5

LAMBDA_MULT = 0.8


DECOMPOSITION_CACHE = os.path.join(cwd, "decomposition_cache.json")


def decompose_once(job_description):
    """Decompose the job description, then reuse that result on every later run.

    The decomposition is a single LLM call and it is NOT reproducible. Two runs
    of the same job description a few hours apart produced 13 groups and then
    12, with one MUST requirement - report the accuracy your models reached -
    silently missing the second time, and vaguer group names throughout. Since
    every score downstream depends on which criteria exist and what they are
    called, that alone makes two runs incomparable.

    The cache is keyed on the job description text, so editing the JD forces a
    fresh decomposition. Delete decomposition_cache.json to force one by hand.
    """
    key = hashlib.sha256(job_description.encode("utf8")).hexdigest()

    if os.path.exists(DECOMPOSITION_CACHE):

        with open(DECOMPOSITION_CACHE, encoding="utf8") as f:
            cached = json.load(f)

        if cached.get("key") == key:
            print("reusing cached decomposition (delete decomposition_cache.json to redo it)")
            return cached["groups"], cached["excluded"]

        print("job description changed, decomposing again")

    groups, excluded = query_decomposition(job_description)

    with open(DECOMPOSITION_CACHE, "w", encoding="utf8") as f:
        json.dump({"key": key, "groups": groups, "excluded": excluded}, f, indent=2)

    return groups, excluded


groups, excluded_requirements = decompose_once(job_description)

if len(groups) == 0:
    raise SystemExit(
        "query_decomposition returned 0 requirement groups. The model did not follow "
        "the GROUP / CRITERION format, so nothing can be retrieved."
    )

criteria_count = sum(len(group["criteria"]) for group in groups)

print(f"{len(groups)} groups, {criteria_count} criteria (decomposed once for the batch)")

if len(excluded_requirements) > 0:
    print(f"{len(excluded_requirements)} requirement(s) skipped, a resume cannot answer them:")
    for requirement in excluded_requirements:
        print(f"  - {requirement}")


PROFILE_CACHE_DIR = os.path.join(cwd, "profile_cache")


def extract_once(path):
    """Extract a resume once, then reuse that JSON on every later run.

    The vision model does not word its output identically on every call - one
    run put a skill in tools_platforms, the next in softwares_used. A different
    word in a chunk changes what retrieval finds, which changes the score, which
    changes the ranking. Caching the extraction removes that entire source of
    drift, and re-running a batch then costs nothing in extraction calls.

    Keyed on the file's own bytes, so editing or replacing a resume re-extracts
    it automatically. Delete profile_cache/ to force a fresh read of everything.
    """
    os.makedirs(PROFILE_CACHE_DIR, exist_ok=True)

    with open(path, "rb") as f:
        key = hashlib.sha256(f.read()).hexdigest()

    cache_file = os.path.join(PROFILE_CACHE_DIR, key + ".json")

    if os.path.exists(cache_file):
        with open(cache_file, encoding="utf8") as f:
            print(f"  reusing cached extraction for {os.path.basename(path)}")
            return json.load(f)

    profile = get_profile(path)

    with open(cache_file, "w", encoding="utf8") as f:
        json.dump(profile, f, indent=2)

    return profile


def screen_resume(path, groups, report_path="output_file.txt",
                  gradesheet_path="gradesheet.txt", verdict_path="verdict.txt"):

    try:
        resume_data = extract_once(path)

    except ExtractionFailed as error:
        print(f"  {error}")
        return {
            "file": os.path.basename(path),
            "name": None,
            "score": None,
            "profile": None,
            "status": "extraction failed - manual review"
        }

    candidate_name = resume_data.get("name") or "unknown"

    chunks = create_structural_chunks_from_resume(resume_data)

    # A profile with nothing in it produces no chunks, and an empty index is a
    # confusing error two layers down in retrieval rather than here. It also
    # must not be scored: no chunks means every criterion comes back NOT_MET
    # and the resume lands at the bottom of the ranking looking like a genuinely
    # weak candidate, when really nothing was ever read.
    if len(chunks) == 0:
        print(f"  nothing readable in {os.path.basename(path)}")
        return {
            "file": os.path.basename(path),
            "name": candidate_name,
            "score": None,
            "profile": None,
            "status": "the extractor returned an empty profile - manual review"
        }

    dense_vector_db = create_embeddings(chunks)
    sparse_db, tfidf = create_sparse_db(chunks)

    evidence = build_evidence(
        groups,
        chunks,
        dense_vector_db,
        sparse_db,
        tfidf,
        resume_file=os.path.basename(path),
        candidate_name=candidate_name,
        top_k=TOP_K,
        final_k=FINAL_K,
        lambda_mult=LAMBDA_MULT,
        output_path=report_path
    )

    final_score, unreadable_requirements, locally_graded_groups = generate_score(
        evidence,
        output_path=gradesheet_path
    )

    status = "ok"
    if len(locally_graded_groups) > 0:
        status = "graded by local model - manual checking required"

    write_verdict(
        evidence,
        resume_file=os.path.basename(path),
        candidate_name=candidate_name,
        final_score=final_score,
        status=status,
        output_path=verdict_path
    )

    return {
        "file": os.path.basename(path),
        "name": candidate_name,
        "score": final_score,
        # carried through so the bias check can group candidates by institution
        # and department without re-reading the extraction cache
        "profile": resume_data,
        "status": status,
        "chunks": len(chunks),
        "unreadable": unreadable_requirements,
        "locally_graded": locally_graded_groups
    }

test_resumes_directory = os.path.join(cwd,"test_resumes")
verdicts_directory = os.path.join(cwd,"test_resumes_verdicts")
# Interview invitations go to these addresses, in ranked order - the top
# candidate gets the first, and the shortlist is however long this list is.
# Put your own addresses here before running.
emails_list = ["eklavyad89@gmail.com", "vrushaligade@gmail.com", "vrushaligade1@gmail.com"]

# How far down the ranking the bias check looks. This is deliberately wider
# than the invitation list: with a big pile of resumes, three selections is far
# too small a sample for a selection ratio to mean anything, since one
# candidate either way swings a group by 33 points. Ten gives the group counts
# enough room to show a real pattern. Nobody outside emails_list is contacted.
BIAS_BAND_SIZE = 10

# What counts as a resume. Everything else in the folder is left alone - the
# reader has no handler for a .gitkeep or a stray .DS_Store and dies on the
# first one it meets, taking the whole batch with it.
RESUME_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg")


def list_resumes(directory):
    """The resume files in a folder, in a fixed order.

    Sorted because os.listdir returns whatever order the filesystem hands
    back. Two candidates on the same score are separated by the order they
    were screened in, so an unsorted listing means the same batch can rank
    them differently on different machines.
    """
    everything = sorted(os.listdir(directory))

    resumes = [
        name for name in everything
        if name.lower().endswith(RESUME_EXTENSIONS)
    ]

    skipped = [name for name in everything if name not in resumes]

    if len(skipped) > 0:
        print(f"ignoring {len(skipped)} non-resume file(s): {', '.join(skipped)}")

    return resumes


# Where the traceback of anything that crashed mid-batch goes. Written fresh
# each run, so what is in it always belongs to the run that just happened.
failures_path = os.path.join(cwd, "screening_failures.txt")


def full_run():

    os.makedirs(verdicts_directory, exist_ok=True)

    if os.path.exists(failures_path):
        os.remove(failures_path)

    filenames = list_resumes(test_resumes_directory)

    if len(filenames) == 0:
        raise SystemExit(
            f"no resumes found in {test_resumes_directory}. Put PDF, PNG or "
            f"JPG files there and run again."
        )

    print(f"screening {len(filenames)} resume(s)")

    results = []

    for filename in filenames:

        try:
            result = screen_resume(
                os.path.join(test_resumes_directory, filename),
                groups,
                report_path=os.path.join(verdicts_directory, filename + "_evidence.txt"),
                gradesheet_path=os.path.join(verdicts_directory, filename + "_gradesheet.txt"),
                verdict_path=os.path.join(verdicts_directory, filename + "_verdict.txt")
            )

        # One bad resume must not take the rest of the batch with it. On a pile
        # of twenty, an unguarded exception on number three throws away the two
        # already screened and never reaches the other seventeen - which is a
        # far worse outcome than one file going to a human.
        #
        # screen_resume already handles the readers all giving up. This catches
        # everything else, and there is a real list of those: a file that is
        # named .pdf but is not one raises FileDataError, an extractor that
        # returns "education": null crashes chunking with a TypeError, one that
        # returns a list of strings where dicts belong raises AttributeError,
        # and a resume with too few real words to index raises "empty
        # vocabulary" out of the sparse database. Catching them one by one only
        # covers the ones already met, so this catches the shape of the problem
        # instead: whatever went wrong, that resume is unscored and flagged.
        except Exception as error:

            print(f"  {filename} crashed: {type(error).__name__}: {error}")

            with open(failures_path, "a", encoding="utf8") as f:
                f.write(f"\n{'=' * 70}\n{filename}\n{'=' * 70}\n")
                f.write(traceback.format_exc())

            result = {
                "file": filename,
                "name": None,
                "score": None,
                "profile": None,
                "status": (
                    f"crashed during screening ({type(error).__name__}) - "
                    f"manual review, full traceback in {os.path.basename(failures_path)}"
                )
            }

        if result["score"] is None:
            print(f"  not ranked: {result['file']} - {result['status']}")

        results.append(result)

    ranked = write_ranking(
        results,
        job_description_file=JOB_DESCRIPTION_FILE,
        shortlist_size=len(emails_list),
        output_path=os.path.join(cwd, "pipeline_ranking.txt")
    )

    print(f"\nranking written to pipeline_ranking.txt ({len(ranked)} scored)")

    provider_summary()

    # Runs on the finished ranking and changes nothing about it. It only writes
    # a report, so a flag here never silently reorders anyone or alters who is
    # emailed.
    run_bias_check(
        ranked,
        band_size=BIAS_BAND_SIZE,
        output_path=os.path.join(cwd, "bias_report.txt"),
        invited_size=len(emails_list)
    )

    top_candidates = []

    for index, result in enumerate(ranked[:len(emails_list)]):
        top_candidates.append({
            "name": result["name"],
            "score": result["score"],
            "email": emails_list[index]
        })

    return top_candidates


top_candidates = full_run()

streamlit_process = subprocess.Popen(
    [
        sys.executable, "-m", "streamlit", "run", "streamlit_slot_selection.py",
        "--server.port", "8501",
        "--server.headless", "true"
    ],
    cwd=cwd
)

start_candidate_workflows(top_candidates, proposed_slots)

print("Done - slot selection page is live on http://localhost:8501")
print("Leave this running so candidates can open their links. Ctrl+C to stop.")

try:
    streamlit_process.wait()

except KeyboardInterrupt:
    streamlit_process.terminate()
