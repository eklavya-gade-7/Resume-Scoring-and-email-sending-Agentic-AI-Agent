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
# Same trap as the Cerebras/Groq one below. ChatGoogleGenerativeAI does NOT
# raise google.genai.errors.APIError - it catches that and re-raises its own
# ChatGoogleGenerativeAIError, which descends from Exception and nothing else.
# The two are unrelated classes, so catching only the SDK one let a Gemini 429
# walk straight past the fallback and kill the group instead of falling
# through to Ollama. It is not exported at package level, hence the deep import.
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_core.exceptions import ContextOverflowError
from httpx import ConnectError as OllamaConnectionError
from ollama import ResponseError

load_dotenv()


# Ordered best-daily-budget first. with_fallbacks is stateless, so it starts at
# the top on EVERY call - a model that was rate limited a minute ago is used
# again the moment it recovers, and nothing has to be reset.
LOCAL_MODEL = "qwen3:0.6b"

GROQ_KEY = os.getenv("GROQ_API_KEY")

# Groq's free limits are counted per model, so every extra model named here is
# another 200,000 tokens and another 1,000 requests for the day on the same
# key. Three of them is 600,000 tokens and 3,000 requests, which is what a
# 20-resume batch actually needs. Read live off the response headers:
# x-ratelimit-limit-requests = 1000, x-ratelimit-limit-tokens = 8000 per minute.
#
# The per-minute token limit is the tight one. At 8,000 tokens a minute and a
# judging prompt of roughly 2,000, one model answers about four groups a minute
# and then starts returning 429. That is a wait of well under a minute, not a
# reason to give up on the model, so the Groq client is told to retry instead of
# falling straight through. It honours the retry-after header the API sends
# back, and costs nothing when nothing is rate limited.
GROQ_RETRIES = 4

# init_chat_model has no "cerebras" provider, so this one is constructed
# directly. ChatCerebras subclasses BaseChatOpenAI, which is why the openai
# exception types appear in the fallback list below.
#
# 1M tokens/day, the widest budget in the chain, but it is a $5 credit trial
# rather than a standing free tier. When the credits run out every call comes
# back 402 payment_required, which is an openai.APIStatusError, which is caught
# below and skipped in silence. provider_summary at the end of a batch is what
# makes that visible.
primary = ChatCerebras(
    model="gpt-oss-120b",
    api_key=os.getenv("CEREBRAS_API_KEY"),
    temperature=0
)

second = init_chat_model(                        # 200K tokens/day, 1000 requests
    "openai/gpt-oss-120b", model_provider="groq",
    api_key=GROQ_KEY, temperature=0, reasoning_effort="low",
    max_retries=GROQ_RETRIES
)

# Groq retired llama-3.3-70b-versatile on 16 August 2026 and named this as its
# replacement. Same key, its own separate 200K tokens and 1000 requests.
#
# reasoning_effort is not a nice-to-have here. This is a reasoning model, and
# left alone it writes out its whole thought process before answering: measured
# on a real judging prompt, 1115 output tokens against 58 for gpt-oss-120b. Out
# of a 200K daily budget that is the difference between 76 groups and 130. The
# grading is a labelling job with the rules already spelled out in the prompt,
# so the thinking buys nothing. "none" and "default" are the only two values
# this model accepts - "low" is a 400.
third = init_chat_model(                         # 200K tokens/day, 1000 requests
    "qwen/qwen3.6-27b", model_provider="groq",
    api_key=GROQ_KEY, temperature=0, reasoning_effort="none",
    max_retries=GROQ_RETRIES
)

fourth = init_chat_model(                        # 200K tokens/day, 1000 requests
    "openai/gpt-oss-20b", model_provider="groq",
    api_key=GROQ_KEY, temperature=0, reasoning_effort="low",
    max_retries=GROQ_RETRIES
)

fifth = ChatGoogleGenerativeAI(                  # different company entirely
    model="gemini-flash-latest",
    google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0
)

# Gemini counts its free quota per model too, so the lite one is a second
# Google budget rather than the same one again. It is a weaker model, which is
# why it sits below flash instead of beside it.
sixth = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
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
    GoogleAPIError,            # gemini, raw SDK path
    ChatGoogleGenerativeAIError,  # gemini, what the LangChain wrapper actually raises
    ContextOverflowError,      # prompt longer than this model's context window
    ResponseError,             # ollama, model missing or crashed
    OllamaConnectionError,     # ollama, daemon not running
)

llm = primary.with_fallbacks(
    [second, third, fourth, fifth, sixth, local],
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


def response_text(response):
    """The reply as one string, whatever shape the provider returned.

    Cerebras and Groq set .content to a plain string. Gemini sets it to a list
    of content blocks - [{"type": "text", "text": "..."}, ...] - and running a
    regex over a list raises "expected string or bytes-like object, got 'list'".

    This only ever bites when the chain falls through to Gemini, which is why
    it stayed hidden until a batch large enough to exhaust Cerebras and both
    Groq budgets. Worse, it is a parse failure AFTER a successful call, so the
    fallback chain cannot rescue it: the model answered perfectly well and our
    own reader choked on the envelope.

    Blocks with no "text" key, such as Gemini's reasoning blocks, are dropped.
    """
    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        parts = []

        for block in content:

            if isinstance(block, str):
                parts.append(block)

            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])

        return "\n".join(parts)

    return str(content)


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


# Which model answered how many groups, across the whole batch. with_fallbacks
# is silent by design: a provider that raises is skipped with no message, so a
# key that is dead on every single call looks exactly like a key that is
# working. That is not a hypothetical - a Cerebras 402 hid behind this for a
# whole 22-resume batch while every call quietly ran on Groq's much smaller
# budget. Counting who actually answered is the cheapest way to see it.
PROVIDER_TALLY = {}

# Every cloud model in the chain, in order, with the name it reports back in
# its response metadata and who it belongs to. Only provider_summary uses this,
# to name the ones that contributed nothing.
CHAIN_MODELS = [
    ("gpt-oss-120b", "Cerebras"),
    ("openai/gpt-oss-120b", "Groq"),
    ("qwen/qwen3.6-27b", "Groq"),
    ("openai/gpt-oss-20b", "Groq"),
    ("gemini-flash-latest", "Google"),
    ("gemini-flash-lite-latest", "Google")
]

# The one worth calling out by name. It leads the chain and has by far the
# largest daily budget, so if it answered nothing then every call landed on a
# much smaller quota and the batch will run out far earlier than it should.
PRIMARY_MODEL = "gpt-oss-120b"


def answered(model_name):
    """How many groups a chain model graded.

    Matched loosely because providers do not always echo the name back exactly
    as it was asked for - a moving alias like gemini-flash-latest is reported
    as whichever build it resolved to that day.
    """
    count = 0

    for name, groups in PROVIDER_TALLY.items():
        if name == model_name or name.startswith(model_name):
            count += groups

    return count


def provider_summary():
    """One block naming who did the work. Costs nothing, prints after a batch."""
    if len(PROVIDER_TALLY) == 0:
        return

    total = sum(PROVIDER_TALLY.values())

    print("\n  groups graded, by model")
    print("  " + "-" * 52)

    for name, count in sorted(PROVIDER_TALLY.items(), key=lambda kv: -kv[1]):
        print(f"    {count:>5}  {count * 100 / total:>5.1f}%   {name}")

    # with_fallbacks skips a failing model in silence, so a key that is dead on
    # every single call looks exactly like a key nothing needed to use. Naming
    # the models that answered nothing is the only way to tell the two apart.
    silent = [
        f"{name} ({owner})"
        for name, owner in CHAIN_MODELS
        if answered(name) == 0
    ]

    if len(silent) > 0:
        print("\n  answered nothing this batch: " + ", ".join(silent))
        print("  That is fine if the batch was small. If it was not, the model")
        print("  is failing on every call and being skipped without a message.")

    if answered(PRIMARY_MODEL) == 0:
        print(
            f"\n  WARNING: {PRIMARY_MODEL} on Cerebras answered nothing. It\n"
            f"  leads the chain and carries the largest daily budget, so every\n"
            f"  call landed on a much smaller quota instead. Cerebras free\n"
            f"  access is $5 of credit that expires 30 days after it is\n"
            f"  granted, and once it is spent every call returns 402. Check\n"
            f"  the billing tab at cloud.cerebras.ai."
        )


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
        answered = parse_verdicts(response_text(response), len(criteria))

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

            PROVIDER_TALLY[model_used] = PROVIDER_TALLY.get(model_used, 0) + 1

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
