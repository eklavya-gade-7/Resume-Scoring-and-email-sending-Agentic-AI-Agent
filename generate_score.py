import os
import re

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_cerebras import ChatCerebras

from groq import RateLimitError, APIStatusError, APIConnectionError
from openai import (
    RateLimitError as OpenAIRateLimitError,
    APIStatusError as OpenAIAPIStatusError,
    APIConnectionError as OpenAIAPIConnectionError,
)
from google.genai.errors import APIError as GoogleAPIError
from langchain_core.exceptions import ContextOverflowError
from httpx import ConnectError as OllamaConnectionError
from ollama import ResponseError

load_dotenv()


# Ordered best-daily-budget first. with_fallbacks is stateless, so it starts at
# the top on EVERY call - a model that was rate limited a minute ago is used
# again the moment it recovers, and nothing has to be reset.
LOCAL_MODEL = "qwen3:0.6b"

# init_chat_model has no "cerebras" provider, so this one is constructed
# directly. ChatCerebras subclasses BaseChatOpenAI, which is why the openai
# exception types appear in the fallback list below.
primary = ChatCerebras(                          # 1M tokens/day
    model="gpt-oss-120b",
    api_key=os.getenv("CEREBRAS_API_KEY"),
    temperature=0
)

second = init_chat_model(                        # 200K/day, separate provider
    "openai/gpt-oss-120b", model_provider="groq",
    api_key=os.getenv("GROQ_API_KEY"), temperature=0, reasoning_effort="low"
)

third = init_chat_model(                         # 100K/day, separate Groq budget
    "llama-3.3-70b-versatile", model_provider="groq",
    api_key=os.getenv("GROQ_API_KEY"), temperature=0
)

fourth = ChatGoogleGenerativeAI(                 # different company entirely
    model="gemini-flash-latest",
    google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0
)

# Last resort. It scored 2/7 on the known-answer set and never produced a
# middle grade, so anything it grades is flagged for manual review rather than
# trusted in a ranking.
local = ChatOllama(model=LOCAL_MODEL, temperature=0)


# Only fall through on "this model is unavailable" errors. Catching everything
# would send a genuine prompt bug quietly down all five models and make it look
# like a rate limit.
FALLBACK_ERRORS = (
    OpenAIRateLimitError,      # cerebras, out of tokens or requests
    OpenAIAPIStatusError,      # cerebras, 5xx and 8192-token context overflow
    OpenAIAPIConnectionError,  # cerebras, network
    RateLimitError,            # groq, out of tokens or requests
    APIStatusError,            # groq, 5xx
    APIConnectionError,        # groq, network
    GoogleAPIError,            # gemini, quota / 4xx / 5xx
    ContextOverflowError,      # prompt longer than this model's context window
    ResponseError,             # ollama, model missing or crashed
    OllamaConnectionError,     # ollama, daemon not running
)

llm = primary.with_fallbacks(
    [second, third, fourth, local],
    exceptions_to_handle=FALLBACK_ERRORS
)


VERDICT_SCORES = {
    "MET": 10.0,
    "MOSTLY_MET": 8.0,
    "PARTIALLY_MET": 4.5,
    "NOT_MET": 0.0
}

# Longest first. Plain substring matching would otherwise find "MET" inside
# "NOT_MET" and grade a failure as a pass.
VERDICT_MATCH_ORDER = (
    "PARTIALLY_MET",
    "MOSTLY_MET",
    "NOT_MET",
    "MET"
)

# A bonus requirement should not sink a candidate who meets every mandatory
# one. The job description separates these itself - "Bonus: proficient in speed
# maths" is not the same demand as "must be proficient in C/C++/Java/Python".
PRIORITY_WEIGHTS = {
    "MUST": 1.0,
    "NICE": 0.6
}


JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("human", """You are grading a candidate's resume against a group of related job
requirements. Every requirement in this group is judged against the same
resume evidence.

Judge ONLY using the resume evidence you are given.

Rules:

- Never invent or assume anything that is not written in the evidence.
- The evidence was found by a search engine, which always returns its closest
  guesses even when nothing relevant exists. Being retrieved means nothing.

- Evidence must be about the requirement's OWN subject. Adjacency is not
  evidence. Knowing a programming language is NOT evidence of operating
  systems, computer networks or databases. A robotics project is NOT evidence
  of database design. Being a strong student is NOT evidence of any specific
  technical subject.

- If nothing in the evidence is actually about the subject being asked about,
  the answer is NOT_MET. Do not award PARTIALLY_MET merely because the
  evidence comes from the same broad technical world.

- A broad field CAN be satisfied by specific techniques or tools that genuinely
  belong to it. OpenCV and YOLO do demonstrate computer vision even if that
  exact phrase never appears. But the technique must actually belong to the
  field being asked about.

- If the requirement states a threshold such as a CGPA, a number of years or a
  graduation year, compare the numbers directly and decide. A requirement with
  different thresholds for different groups, for example "7.00 in Computer
  Science branches or 8.00 in all other branches", means the candidate's own
  branch selects which threshold applies. A branch that is not the named one
  falls under "all other branches".

- If the requirement allows alternatives, satisfying one of them is enough.

- Each piece of evidence is labelled with the resume section it came from.
  Weigh the sections differently:
    experience and projects  - strong evidence of practical, hands-on ability
    technical_skills         - proves the candidate lists the skill, but on its
                               own is weak evidence of hands-on experience
    relevant_courses         - proves academic exposure only, weaker still
    education                - use for degree, branch, CGPA and graduation year

- If only part of a requirement is supported, do not award MET.

- The requirements are separate from each other. Grade each one on its own
  merits. Evidence that satisfies requirement 1 does not automatically say
  anything about requirement 2.


Answer each requirement with EXACTLY ONE of these labels:

MET
    The evidence clearly and directly satisfies the requirement.

MOSTLY_MET
    Substantially satisfied, but one minor aspect is missing or unclear.

PARTIALLY_MET
    Genuine evidence about this subject exists, but it is incomplete, indirect
    or weak. A relevant course with no project or work applying it belongs
    here, not at MET.

NOT_MET
    The evidence does not support the requirement. Use this whenever a
    required skill, technology, subject, degree or threshold is absent, and
    whenever the evidence is about something else entirely.
    
Examples:

Example 1:
Requirement: Practical experience with computer vision.
Evidence: Project using YOLOv8 and OpenCV for object detection.
Verdict: MET

Example 2:
Requirement: Practical understanding of operating systems.
Evidence: Programming languages: Python, C++.
Verdict: NOT_MET

Example 3:
Requirement: Practical experience with databases.
Evidence: Relevant coursework includes Database Management Systems.
Verdict: PARTIALLY_MET

Example 4:
Requirement: Proficiency in Python OR Java.
Evidence: Programming languages: Python, C++.
Verdict: MET

Example 5:
Requirement: CGPA >= 8.0 for non-CS branches.
Evidence: Mechanical Engineering, CGPA 7.8.
Verdict: NOT_MET


Output one line per requirement, using the requirement's number:

1: LABEL
2: LABEL

Give a line for every requirement, in order, and output nothing else.
No reasoning, no explanation, no extra words.

RESUME EVIDENCE:

{evidence}


REQUIREMENTS TO GRADE:

{criteria}

"""
)])


# No StrOutputParser here on purpose. The parser throws away
# response_metadata, and that is the only place the name of the model that
# actually answered is recorded - which is how a local-model grade gets flagged.
judge_chain = JUDGE_PROMPT | llm


def render_evidence(chunks):
    """Lay out the group's chunks as numbered, section-labelled blocks.

    The section label is what lets the grading rules treat a skills-list
    mention differently from a described project.
    """
    blocks = []

    for number, chunk in enumerate(chunks, start=1):

        blocks.append(
            f"[EVIDENCE {number} - from the "
            f"{chunk['section']} section of the resume]\n"
            + chunk["text"]
        )

    return "\n\n".join(blocks)


def render_criteria(criteria):
    """Number the group's criteria so the reply can be matched back by index."""
    lines = []

    for number, criterion in enumerate(criteria, start=1):
        lines.append(f"{number}. {criterion}")

    return "\n".join(lines)


def parse_verdicts(response, criteria_count):
    """Read the reply into {criterion number: verdict}.

    A missing number is simply absent from the result rather than guessed at,
    and the caller records it as unreadable instead of scoring it.
    """
    text = re.sub(r"<think>.*?</think>", " ", response, flags=re.S | re.I)

    verdicts = {}

    for line in text.splitlines():

        # Leading junk is skipped on purpose. A model that answers "**1: MET**"
        # or "REQUIREMENT 1: MET" is still answering, and refusing to read
        # those lines would throw away a whole group's worth of work.
        match = re.match(r"[^0-9A-Za-z]*(\d+)\s*[:.\)\-]\s*(.+)", line)

        if match is None:
            continue

        number = int(match.group(1))

        if number < 1 or number > criteria_count:
            continue

        # a repeated number is the model restating itself, not changing its
        # mind, so the first answer wins
        if number in verdicts:
            continue

        verdict = match_verdict(match.group(2))

        if verdict is not None:
            verdicts[number] = verdict

    return verdicts


def match_verdict(text):
    """Find a known label inside one line of the reply, or None."""
    text = re.sub(r"[^A-Za-z]+", "_", text).upper()

    for verdict in VERDICT_MATCH_ORDER:
        if verdict in text:
            return verdict

    return None


def model_name_of(response):
    """Which model actually answered. Providers use different metadata keys."""
    meta = getattr(response, "response_metadata", {}) or {}

    for key in ("model_name", "model", "model_id"):
        if meta.get(key):
            return str(meta[key])

    return "unknown"


def judge_group(group):
    """Grade one group in a single call.

    Returns (verdicts, calls_made, model_used) where verdicts maps each
    criterion to a label or None.
    """
    criteria = group["criteria"]

    results = {}

    # Nothing retrieved at all means nothing in the resume came close. That is
    # a real miss, and it needs no model call.
    if len(group["chunks"]) == 0:

        for criterion in criteria:
            results[criterion] = "NOT_MET"

        return results, 0, "none (no evidence)"

    model_used = "unknown"

    try:
        response = judge_chain.invoke({
            "evidence": render_evidence(group["chunks"]),
            "criteria": render_criteria(criteria)
        })

        model_used = model_name_of(response)
        answered = parse_verdicts(response.content, len(criteria))

    except Exception as error:
        print(f"  grading failed for group '{group['name']}': {error}")
        answered = {}
        model_used = "failed"

    for number, criterion in enumerate(criteria, start=1):
        results[criterion] = answered.get(number)

    return results, 1, model_used


def generate_score(evidence, output_path="gradesheet.txt"):
    """Grade every group and return (final_score, unreadable_requirements).

    evidence is the list of groups from build_evidence. One model call is made
    per group, but every criterion inside the group is scored separately, so
    grouping reduces calls without reducing scoring detail.

    Every criterion reaching this function is one a resume can answer - the
    rest were dropped during decomposition. So missing evidence means the
    candidate lacks it, and NOT_MET scoring zero is correct.
    """
    score = 0.0
    total_possible_score = 0.0

    unreadable_requirements = []
    locally_graded_groups = []
    api_calls = 0

    with open(output_path, "w", encoding="utf8") as f:

        f.write("SUBQUERY : VERDICT : SCORE ASSIGNED\n")
        f.write("=" * 100 + "\n\n")

        for group in evidence:

            priority = group["priority"]

            weight = PRIORITY_WEIGHTS.get(priority, 1.0)
            possible = 10.0 * weight

            verdicts, calls_made, model_used = judge_group(group)
            api_calls += calls_made

            # LOCAL_MODEL only runs when every cloud model is exhausted. It
            # scored 2/7 on the known-answer set, so its grades are marked
            # instead of being quietly trusted in a ranking.
            graded_locally = LOCAL_MODEL in model_used

            if graded_locally:
                locally_graded_groups.append(group["name"])

            note = ""
            if graded_locally:
                note = "   *** GRADED BY LOCAL MODEL - MANUAL CHECKING REQUIRED ***"

            f.write(f"--- GROUP: {group['name']}  [{priority}]  (model: {model_used}){note}\n")

            for criterion in group["criteria"]:

                verdict = verdicts[criterion]

                # A model that returns gibberish should not cost the candidate
                # points, so an unreadable reply is excluded rather than
                # scored zero.
                if verdict is None:
                    unreadable_requirements.append(criterion)
                    f.write(
                        f"{criterion} : UNREADABLE_MODEL_REPLY :"
                        f" excluded from scoring\n"
                    )
                    continue

                awarded = VERDICT_SCORES[verdict] * weight

                score += awarded
                total_possible_score += possible

                f.write(
                    f"{criterion} : {verdict} : {awarded:.2f} / {possible:.2f}"
                    f"  ({priority})\n"
                )

            f.write("\n")

        if total_possible_score == 0:
            final_score = 0.0
        else:
            final_score = (score * 100.0) / total_possible_score

        criteria_count = sum(len(group["criteria"]) for group in evidence)
        graded = criteria_count - len(unreadable_requirements)

        f.write("=" * 100 + "\n")
        f.write(f"FINAL SCORE : {final_score:.2f} / 100\n")
        f.write(f"POINTS      : {score:.2f} / {total_possible_score:.2f}\n")
        f.write(f"GRADED      : {graded} of {criteria_count} criteria\n")
        f.write(
            f"LLM CALLS   : {api_calls} for {len(evidence)} group(s)"
            f" covering {criteria_count} criteria\n"
        )

        if len(locally_graded_groups) > 0:
            f.write("\n*** MANUAL CHECKING REQUIRED ***\n")
            f.write("Every cloud model was exhausted, so these groups were graded by the\n")
            f.write(f"local {LOCAL_MODEL} model. Its verdicts are not reliable - re-check them\n")
            f.write("before using this score in a ranking.\n")
            for name in locally_graded_groups:
                f.write(f"  - {name}\n")

        if len(unreadable_requirements) > 0:
            f.write("\nMODEL REPLY COULD NOT BE READ (excluded from the score)\n")
            for criterion in unreadable_requirements:
                f.write(f"  - {criterion}\n")

    return final_score, unreadable_requirements, locally_graded_groups
