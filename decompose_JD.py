import langchain
import langchain_groq
import os
from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
import contractions
import re

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# 8b-instant at temperature 0.3 kept ignoring the output format - it added
# "Here are the requirements:" preambles, numbered every line, and invented a
# fourth synonyms field. Every one of those made the parser drop the line. This
# used to be llama-3.3-70b-versatile at temperature 0, which Groq retired on
# 16 August 2026. gpt-oss-120b is the bigger of the two replacements Groq names,
# and the same model generate_score.py already grades with.
#
# This one call has no fallback behind it, so a dead model name here stops the
# whole pipeline at import. The result is cached in decomposition_cache.json,
# so an existing cache keeps its old groups and scores stay comparable.
model = init_chat_model("openai/gpt-oss-120b", api_key=GROQ_API_KEY, temperature=0, model_provider="groq")

decomposition_prompt = PromptTemplate.from_template(
"""
You are analyzing a job description for resume matching.

Your task is to break the job description into independent, important,
atomic candidate requirements that can later be used to evaluate a resume.

Each requirement should represent exactly ONE meaningful qualification,
skill, experience, capability, educational condition, or eligibility
criterion required or preferred by the job description.


==================================================
1. PRIORITY CLASSIFICATION
==================================================

Classify every requirement as exactly one of:

MUST
NICE


MUST:

Use MUST when the job description states or strongly implies that the
requirement is required, mandatory, a minimum qualification, an eligibility
condition, or a core capability expected from the candidate.


NICE:

Use NICE when the job description describes the requirement as preferred,
desirable, optional, advantageous, "a plus", "good to have", or something
that strengthens the candidate without being mandatory.


IMPORTANT:

MUST/NICE describes how IMPORTANT the requirement is.

It is completely independent of the answerability decision and the retrieval
classification described below.


==================================================
2. WHAT MUST NEVER BECOME A REQUIREMENT
==================================================

Some things in a job description must NEVER be turned into a requirement,
because a resume cannot answer them. Produce no line at all for these.

Eligibility and logistics conditions:

- active backlogs or arrears
- work authorization, visa status or sponsorship
- willingness to relocate
- joining availability, notice period or joining date
- willingness to work particular shifts
- willingness to travel
- background checks or security clearance
- service bonds

Things the job OFFERS the candidate rather than demands from them:

- salary, stipend or compensation
- benefits, perks, insurance, accommodation or meals
- job location or office address
- company history, size, funding or reputation
- training, mentorship or team environment offered

Skip all of the above silently. Do not write a line for them in any form,
and do not mention that you skipped them.

Every requirement you DO produce must be something a resume can answer:
skills, degree, branch, CGPA, graduation year, coursework, projects,
experience, internships, certifications, achievements, competitions or
leadership roles.


==================================================
3. RETRIEVAL CLASSIFICATION
==================================================

Classify every requirement as exactly one of:

DENSE
SPARSE
HYBRID


------------------------------
DENSE
------------------------------

Classify the requirement as DENSE when semantic or meaning-oriented
retrieval is especially important for finding relevant resume evidence.

Use DENSE when the requirement may be demonstrated in a resume using
different terminology, implementations, achievements, responsibilities,
subtopics, techniques, or descriptions.

Keyword matching alone may therefore fail to retrieve the best evidence.

DENSE commonly applies to:

- broad technical domains
- conceptual understanding
- general areas of expertise
- responsibilities
- capabilities
- achievements
- problem-solving experience
- system-design experience
- domain experience
- experience that can be demonstrated through multiple different techniques
  or implementations


------------------------------
SPARSE
------------------------------

Classify the requirement as SPARSE when matching primarily depends on the
presence of a particular named term or one of a clearly specified set of
acceptable terms.

SPARSE commonly applies to:

- programming languages
- frameworks
- libraries
- databases
- tools
- technologies
- certifications
- degrees
- departments / branches / majors
- explicitly named methodologies
- CGPA / GPA / percentage conditions
- graduation years
- other specific qualifications where the exact or near-exact term matters

Do NOT classify something as SPARSE merely because it is mandatory.


------------------------------
HYBRID
------------------------------

Classify the requirement as HYBRID when BOTH of the following matter:

1. A specific named technology, skill, tool, qualification, platform,
   framework, language, or other term matters.

AND

2. Broader semantic experience, responsibility, knowledge, or capability
   involving that term also matters.

HYBRID should therefore benefit from both keyword-oriented retrieval and
meaning-oriented retrieval.


==================================================
4. REQUIREMENT QUESTION
==================================================

Write each requirement as a short and clear QUESTION about the candidate.

The question should be suitable for later grading against resume evidence.

For example:

"Does the candidate have experience with PostgreSQL?"

"Does the candidate have practical knowledge of operating systems?"

"Does the candidate have at least 3 years of backend development experience?"

"Does the candidate have the required degree?"

Keep the question focused and concise.


------------------------------
ONE CRITERION TESTS ONE THING
------------------------------

A criterion must be something that can be true or false on its own.

If a job description names several subjects, technologies or skills that are
ALL expected, write ONE CRITERION FOR EACH. Never join them into a single
question using "and".

WRONG - four subjects crushed into one question:

  "Does the candidate have a practical understanding of CS concepts in the
   areas of Data Structures, Operating Systems, Computer networks and
   Database management systems?"

RIGHT - four separate criteria:

  "Does the candidate have a practical understanding of Data Structures and
   algorithms?"
  "Does the candidate have a practical understanding of Operating Systems?"
  "Does the candidate have a practical understanding of Computer networks?"
  "Does the candidate have a practical understanding of Database management
   systems?"

A candidate can know Operating Systems and not know Databases. The joined
question cannot record that difference. The four separate criteria can.

The same applies to a bonus line such as "speed maths and probability" or
"Machine Learning and Computer Vision". Those are separate skills and need
separate criteria.


The ONLY time several named things belong inside ONE criterion is when the
job description says any ONE of them is enough.

So the test is:

  joined by OR, "one or more of", "any of"   ->  ONE criterion
  joined by AND, or a list of expected areas ->  ONE CRITERION FOR EACH


------------------------------
ALTERNATIVE REQUIREMENTS
------------------------------

If the job description accepts one of multiple alternatives, preserve those
alternatives inside ONE requirement.

For example, if the job description says:

"Proficiency in Python, C++, or Java"

do NOT produce three mandatory requirements.

Instead produce one requirement asking whether the candidate has proficiency
in one or more of the accepted languages.

Similarly, correctly preserve alternatives involving:

- degrees
- branches
- technologies
- certifications
- platforms
- experience types
- or any other requirement where the job description allows multiple options


==================================================
5. RETRIEVAL QUERY EXPANSION
==================================================

For every group, generate one or more RETRIEVAL QUERIES.

One search is run per group, so these queries must be able to find evidence
for EVERY requirement inside that group.

These retrieval queries will be used ONLY to search the candidate's resume.

They are NOT additional requirements.

They must NEVER change what the job description actually requires.

The ORIGINAL REQUIREMENT QUESTION will later be used for grading.

The RETRIEVAL QUERIES are only different ways of finding potentially
relevant evidence inside the resume.


------------------------------
DENSE QUERY EXPANSION
------------------------------

For a DENSE requirement, identify whether the requirement contains a broad
domain, umbrella concept, capability, field, responsibility, or area of
knowledge.

If it does, generate several focused retrieval queries representing the
major subtopics, tasks, techniques, implementations, responsibilities, or
concepts through which a candidate could reasonably demonstrate that broad
requirement.

This is important because a candidate may have strong evidence for a broad
area without explicitly writing the broad umbrella term in the resume.

Generate approximately 2 to 5 focused retrieval queries when expansion is
useful.

Do not simply generate synonyms.

Instead generate different realistic evidence routes.

Do not generate unrelated technologies merely because they are sometimes
associated with the field.


------------------------------
SPARSE QUERY EXPANSION
------------------------------

For a SPARSE requirement, keep the retrieval queries narrow.

Use:

- the exact required term
- obvious abbreviations
- obvious spelling variants
- common aliases
- explicitly accepted alternatives mentioned in the job description

Do NOT replace a specifically required technology with unrelated alternatives.

For example, if PostgreSQL is specifically required, do not add MySQL,
MongoDB, or unrelated databases.


------------------------------
HYBRID QUERY EXPANSION
------------------------------

For a HYBRID requirement:

- preserve the important exact named term
- also generate a small number of meaning-oriented retrieval queries
  describing the broader experience or capability involving that term

The exact required technology or term must remain represented in the
retrieval queries.


==================================================
6. QUERY EXPANSION EXAMPLES
==================================================

The following examples demonstrate the GENERAL REASONING PATTERN.

They are examples only.

Do NOT copy these requirements or subtopics unless they are relevant to the
actual job description.


------------------------------
EXAMPLE A: BROAD COMPUTER VISION REQUIREMENT
------------------------------

Requirement:

Does the candidate have experience in computer vision?

Classification:

DENSE

Possible retrieval queries:

computer vision
||
object detection and visual recognition
||
image processing and image analysis
||
OCR and image understanding
||
image classification and image segmentation

Reasoning pattern:

A candidate might demonstrate strong computer-vision experience through
YOLO, R-CNN, OCR, image segmentation, object detection, OpenCV-based work,
or other related projects without explicitly writing the phrase
"computer vision".


------------------------------
EXAMPLE B: BROAD OPERATING SYSTEMS REQUIREMENT
------------------------------

Requirement:

Does the candidate have practical knowledge of operating systems?

Classification:

DENSE

Possible retrieval queries:

operating systems
||
processes and threads
||
CPU scheduling and concurrency
||
synchronization and deadlocks
||
memory management and virtual memory
||
file systems and system calls

Reasoning pattern:

A candidate may demonstrate operating-systems knowledge through specific
topics or projects involving processes, threads, scheduling, synchronization,
memory management, or other OS concepts without explicitly writing the
phrase "operating systems".


------------------------------
EXAMPLE C: NARROW NAMED TECHNOLOGY
------------------------------

Requirement:

Does the candidate have experience with PostgreSQL?

Classification:

SPARSE

Possible retrieval queries:

PostgreSQL
||
Postgres

Reasoning pattern:

PostgreSQL is specifically required.

Do NOT expand this into MySQL, MongoDB, SQL Server, or other databases that
were not accepted by the job description.


------------------------------
EXAMPLE D: NAMED TECHNOLOGY + BROADER EXPERIENCE
------------------------------

Requirement:

Does the candidate have experience deploying machine learning systems on AWS?

Classification:

HYBRID

Possible retrieval queries:

AWS
||
deploying machine learning models on AWS
||
cloud deployment of machine learning systems
||
production machine learning infrastructure on AWS

Reasoning pattern:

AWS is an exact named technology that matters, while deployment experience
is broader and may be described in many different ways.


==================================================
7. IMPORTANT GENERAL RULES
==================================================

- Produce only requirements that meaningfully affect whether the candidate
  is suitable or eligible for the job.

- Do not invent requirements that are not present in the job description.

- Do not create duplicate or substantially overlapping requirements.

- Do not combine unrelated requirements into one query.

- One query should correspond to one meaningful requirement.

- Preserve explicit quantitative conditions.

Examples:

"3+ years of experience"
"CGPA of at least 8.0"
"Graduating in 2027"
"Minimum 60 percent"
"At least 2 years of Python experience"

If such a condition is explicitly present, retain it.

Do NOT invent numerical requirements that were not stated.


- Pay particular attention to resume-relevant candidate information such as:

  programming languages
  technical skills
  frameworks
  libraries
  databases
  tools
  technologies
  cloud platforms
  degree
  department / branch / major
  CGPA / GPA / percentage
  graduation year
  professional experience
  internships
  years of experience
  technical domains
  projects
  relevant coursework
  certifications
  achievements
  competitions
  leadership and positions of responsibility


- Ignore information describing what the JOB or COMPANY gives to the
  candidate.

Examples include:

  salary
  employee benefits
  learning opportunities
  exposure to experienced professionals
  company culture
  mentorship offered
  team environment
  perks
  corporate experience offered by the internship


- Ignore statements such as:

  "You will work with an experienced team of industry professionals."

  "You will get exposure to real-world projects."

unless such a statement contains an actual qualification required from the
candidate.


- Job location is not a resume-grading requirement, and neither is a
  condition such as "Candidate must be willing to relocate". Both belong to
  the skip list in section 2.


- Do not classify something as SPARSE merely because it is MUST.

- Do not classify something as DENSE merely because it is NICE.

Priority classification and retrieval classification are independent.


==================================================
8. QUERY EXPANSION SAFETY RULES
==================================================

The expanded retrieval queries exist only to FIND possible resume evidence.

They must NEVER:

- create new job requirements
- make the original requirement stricter
- make the original requirement weaker
- assume that a candidate possesses a generated subtopic
- be treated as evidence themselves


When expanding a broad requirement, think about:

- subtopics
- tasks
- techniques
- implementations
- responsibilities
- concepts
- algorithms
- systems
- realistic project work

that could constitute credible resume evidence for that broad requirement.


Do NOT merely generate synonyms of the broad term.


For example, a useful broad-domain expansion looks conceptually like:

umbrella domain
→ important subtopic
→ realistic implementation/task
→ possible evidence in resume

rather than:

umbrella term
→ synonym
→ synonym
→ synonym


==================================================
9. GROUPING REQUIREMENTS
==================================================

Requirements that the SAME lines of a resume would answer are searched for
and graded together, as one GROUP.

Grouping changes how many searches are run. It does NOT change how many
things get judged. Every requirement inside a group still receives its own
separate verdict, so a candidate who has AutoCAD but not SolidWorks is still
recorded as having one and not the other.


------------------------------
WHEN TO GROUP
------------------------------

Put two requirements in the same group ONLY when BOTH of these are true:

1. They concern the same underlying competency.
2. The same lines of a resume would very likely answer both.

Aim for at most 6 groups. This is a target, not a hard limit, and it must
never be reached by merging requirements together. Reducing the number of
groups is worthwhile. Reducing the number of criteria is not - every
requirement in the job description still needs its own criterion. A job
description containing many genuinely independent requirements should produce
more than 6 groups rather than force unrelated things together.

A group holding a single requirement is normal and correct. Never invent a
grouping just to reduce the number of groups.


------------------------------
WHAT MUST SURVIVE GROUPING
------------------------------

- Keep every atomic requirement. Grouping must never merge two requirements
  into one, and must never drop one.
- Never lose an explicitly named technology, threshold, qualification or
  condition.
- Preserve AND / OR relationships exactly as the job description states them.
- Never put a MUST requirement and a NICE requirement in the same group.
- Never group requirements merely because they belong to the same broad field.


------------------------------
GROUPING EXAMPLES
------------------------------

GOOD. The same resume lines would answer all three, so one group holding
three separate requirements:

  AutoCAD
  SolidWorks
  CATIA

  -> GROUP "Mechanical CAD tools", with three criteria inside it.


GOOD. The job description itself says any one of them is enough, so this was
already a single requirement before grouping:

  Python OR C++ OR Java

  -> GROUP holding ONE criterion asking about one or more of the accepted
     languages.


BAD. These share a broad field, but a candidate can easily know one and not
the others, and completely different resume lines would prove each one:

  Operating Systems
  Computer Networks
  Databases

  -> THREE separate groups, each holding its own criterion. Do NOT create a
     single "CS fundamentals" group, and do NOT write one criterion that asks
     about all three at once.


==================================================
10. OUTPUT FORMAT
==================================================

Return each group as ONE "GROUP" line, followed by one "CRITERION" line for
every requirement inside that group.

GROUP | PRIORITY | RETRIEVAL_TYPE | GROUP NAME | QUERY_1 || QUERY_2 || QUERY_3 ...
CRITERION | requirement question
CRITERION | requirement question

PRIORITY and RETRIEVAL_TYPE describe the whole group.

The retrieval queries describe the whole group, because one search is run per
group rather than one per criterion.

Every GROUP line must be followed by at least one CRITERION line.


Examples of FORMAT ONLY:

GROUP | MUST | SPARSE | Named tool family | first tool || second tool || third tool
CRITERION | Does the candidate have experience with the first named tool?
CRITERION | Does the candidate have experience with the second named tool?
CRITERION | Does the candidate have experience with the third named tool?

GROUP | MUST | DENSE | Broad technical domain | broad domain || important subtopic || related implementation
CRITERION | Does the candidate have practical knowledge of the required broad technical domain?

GROUP | NICE | HYBRID | Preferred technology experience | specified technology || relevant experience using specified technology
CRITERION | Does the candidate have the preferred experience involving the specified technology?


The examples above demonstrate only the required FORMAT and reasoning
behaviour.

Do NOT copy their requirements, technologies, or retrieval queries unless
they actually correspond to the supplied job description.


Do not return:

- explanations
- reasoning
- numbering
- headings
- JSON
- markdown
- introductory text
- concluding text

Return ONLY the formatted GROUP and CRITERION lines.


==================================================
JOB DESCRIPTION
==================================================

{context}
"""
)






# Conditions a resume does not carry as a document, no matter how strong the
# candidate is. The prompt already tells the model to skip these, but at
# temperature 0 Groq is still not bit-deterministic, so one of them slips
# through every few runs. This is the deterministic backstop.
#
# Unlike a skills list this set is closed. It describes what the resume FORMAT
# leaves out, which is roughly a dozen things and barely changes between job
# descriptions - not what technologies exist in the world, which is unbounded.
#
# Terms are kept specific on purpose. Bare "shift" would catch "shift register"
# and bare "bond" would catch "wire bonding" or "chemical bonding", both of
# which are real requirements on a mechanical or electrical resume.
NOT_IN_RESUME_TERMS = (
    "backlog",
    "arrear",
    "relocat",
    "notice period",
    "work authorization",
    "work authorisation",
    # not bare "visa" - that matches Visa the payments company, which is a
    # genuine requirement on a fintech job description
    "visa status",
    "visa sponsorship",
    "work visa",
    "valid visa",
    "sponsorship",
    "security clearance",
    "background check",
    "willing to travel",
    "willingness to travel",
    "night shift",
    "rotational shift",
    "shift work",
    "work in shifts",
    "working in shifts",
    "joining date",
    "date of joining",
    "availability to join",
    "immediate joiner",
    "service bond",
    "employment bond",
)


def is_answerable_from_resume(subquery):
    """False when the requirement asks about something resumes do not state.

    Such requirements are dropped rather than graded. Scoring one as a miss
    would penalise the candidate for the document format instead of for
    anything about the candidate.
    """
    lowered = subquery.lower()

    for term in NOT_IN_RESUME_TERMS:
        if term in lowered:
            return False

    return True


def parse_group_line(line):
    """Read a GROUP header into a group dict, or None if it is unusable.

    GROUP | PRIORITY | RETRIEVAL_TYPE | GROUP NAME | QUERY_1 || QUERY_2
    """
    parts = line.split("|", 4)

    if len(parts) < 4:
        return None

    priority = parts[1].strip().upper()
    retrieval_type = parts[2].strip().upper()
    name = parts[3].strip()

    # A misspelled priority must not delete the requirement. MUST is the safe
    # default - treating a mandatory requirement as optional is the worse of
    # the two mistakes for a screening tool.
    if priority not in ("MUST", "NICE"):
        priority = "MUST"

    if retrieval_type not in ("DENSE", "SPARSE", "HYBRID"):
        retrieval_type = "HYBRID"

    retrieval_queries = []

    if len(parts) == 5:

        for query in parts[4].split("||"):

            query = query.strip()

            if query != "" and query.upper() != "NONE":
                retrieval_queries.append(query)

    return {
        "name": name,
        "priority": priority,
        "retrieval_type": retrieval_type,
        "retrieval_queries": retrieval_queries,
        "criteria": []
    }


def query_decomposition(job_description):
    """Break a job description into requirement groups.

    Returns (groups, excluded_requirements) where each group is

        {"name", "priority", "retrieval_type", "retrieval_queries", "criteria"}

    One search is run per group, but every criterion inside it is still graded
    separately - grouping reduces the number of searches and model calls, not
    the number of scoring decisions.
    """
    query_decomposition_chain = create_stuff_documents_chain(model,decomposition_prompt)

    job_description = contractions.fix(job_description)
    jd_document = Document(page_content = job_description)

    response = query_decomposition_chain.invoke({"context" : [jd_document]})

    groups = []
    excluded_requirements = []

    current_group = None

    for line in response.split("\n"):

        # Models like to decorate their output ("1. GROUP", "- GROUP",
        # "**GROUP"). Anything before the first letter is formatting.
        line = re.sub(r"^[^A-Za-z]+", "", line.strip())

        if line == "":
            continue

        if line.upper().startswith("GROUP"):

            current_group = parse_group_line(line)

            if current_group is not None:
                groups.append(current_group)

            continue

        if line.upper().startswith("CRITERION"):

            # a criterion before any group header has nowhere to belong
            if current_group is None:
                continue

            parts = line.split("|", 1)

            if len(parts) < 2:
                continue

            criterion = parts[1].strip()

            if criterion == "":
                continue

            # Drop it instead of grading it. Keeping these as gradeable
            # requirements is what let the judge confuse "a resume never states
            # this" with "the candidate lacks this skill".
            if not is_answerable_from_resume(criterion):
                excluded_requirements.append(criterion)
                continue

            current_group["criteria"].append(criterion)

    # a group whose criteria were all dropped has nothing left to grade
    groups = [
        group
        for group in groups
        if len(group["criteria"]) > 0
    ]

    return groups, excluded_requirements
