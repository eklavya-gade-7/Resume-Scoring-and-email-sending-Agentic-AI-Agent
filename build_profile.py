from pydantic import BaseModel, Field, ValidationError
from langchain_core.exceptions import OutputParserException
from groq import RateLimitError, APIStatusError, APIConnectionError
from openai import (
    RateLimitError as OpenAIRateLimitError,
    APIStatusError as OpenAIAPIStatusError,
    APIConnectionError as OpenAIAPIConnectionError,
)
from google.genai.errors import APIError as GoogleAPIError
from langchain_core.exceptions import ContextOverflowError
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Grading and parsing run on separate daily budgets, so each step keeps its own
# ordered list of models. The first one that answers wins; if it is rate limited
# or the resume is too long for its context window, the next one is tried.
#
# with_fallbacks is stateless - it starts from the top on EVERY call. So a model
# that was exhausted a minute ago is used again the moment it recovers, and
# nothing needs resetting.
def build_model_chain():
    """Ordered list of models for markdown -> JSON, best budget first."""
    models = []

    # Cerebras gives 1M tokens/day free, far more than Groq, so it goes first
    # once the key exists. Its free tier caps context at 8192 tokens and this
    # call is ~6800, so a long resume will overflow - that raises, and the next
    # model in the list picks it up.
    if os.getenv("CEREBRAS_API_KEY"):
        try:
            from langchain_cerebras import ChatCerebras
            models.append(ChatCerebras(
                # llama-3.3-70b returned 404 "model does not exist or you do not
                # have access to it" on this key. gpt-oss-120b is the Cerebras
                # model the grader already uses successfully.
                model="gpt-oss-120b",
                api_key=os.getenv("CEREBRAS_API_KEY"),
                temperature=0
            ))
        except ImportError:
            print("CEREBRAS_API_KEY set but langchain-cerebras is not installed")

    # known good with this exact nested schema
    models.append(init_chat_model(
        "groq:llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0
    ))

    # different Groq model, so a separate daily token budget
    models.append(init_chat_model(
        "openai/gpt-oss-120b",
        api_key=GROQ_API_KEY,
        temperature=0,
        model_provider="groq"
    ))

    # different provider entirely, and it has native structured output
    if os.getenv("GEMINI_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        models.append(ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0
        ))

    return models


MODEL_CHAIN = build_model_chain()

model = MODEL_CHAIN[0]


class Experience(BaseModel):
    company: str | None = Field(
        default=None,
        description="Name of the company or organization where the candidate worked."
    )

    job_title: str | None = Field(
        default=None,
        description="Job title or internship position held by the candidate."
    )

    start: str | None = Field(
        default=None,
        description="Start date as YYYY-MM or YYYY if month is unavailable. Null if not stated."
    )

    end: str | None = Field(
        default=None,
        description="End date as YYYY-MM or YYYY. Use null for Present, ongoing or currently working."
    )

    technologies: list[str] = Field(
        default_factory=list,
        description="Tools, languages, frameworks, libraries and technologies explicitly "
                    "mentioned as being used in this role, including inside bullet points."
    )

    bullets: list[str] = Field(
        default_factory=list,
        description="Every bullet point describing this role, copied word for word."
    )


class Education(BaseModel):
    institution: str | None = Field(
        default=None,
        description="Institution corresponding to this qualification. Null if not stated."
    )

    degree: str | None = Field(
        default=None,
        description="Qualification such as B.Tech, MSc, PhD, Class XII or diploma."
    )

    field: str | None = Field(
        default=None,
        description="Department, branch, major, subject or field of study such as "
                    "Computer Science, Mechanical Engineering or Electrical Engineering."
    )

    year: str | int | None = Field(
        default=None,
        description="Year of completion or expected completion. Null if not stated."
    )

    cgpa: str | None = Field(
        default=None,
        description="CGPA, GPA, percentage, marks or grade copied exactly as printed, "
                    "for example '8.36', '90.8%' or '3.8/4.0'. Null if absent."
    )


class Project(BaseModel):
    name: str | None = Field(
        default=None,
        description="Name of the personal, academic, team or organizational project."
    )

    technologies: list[str] = Field(
        default_factory=list,
        description="Programming languages, frameworks, libraries, tools and technologies "
                    "explicitly mentioned in this project."
    )

    bullets: list[str] = Field(
        default_factory=list,
        description="Every project description bullet copied word for word."
    )


class Position(BaseModel):
    role: str | None = Field(
        default=None,
        description="Position held, such as Events Manager, Captain, Coordinator, Deputy "
                    "Coordinator, Manager, Lead, Mentor or Project Member. If the resume "
                    "states no title and only says the candidate was part of the group, "
                    "use 'Member'. If it only says they volunteered, use 'Volunteer'."
    )

    organization: str | None = Field(
        default=None,
        description="Club, student team, society, committee, chapter, cell, contingent, "
                    "fest, council, chapter or organization associated with the position."
    )

    bullets: list[str] = Field(
        default_factory=list,
        description="Every line describing work, leadership or achievements in this position."
    )


class TechnicalSkills(BaseModel):
    programming_languages: list[str] = Field(
        default_factory=list,
        description="Programming languages explicitly mentioned, such as Python, C++, "
                    "Java, JavaScript, SQL or MATLAB."
    )

    frameworks_libraries: list[str] = Field(
        default_factory=list,
        description="Frameworks and libraries explicitly mentioned, such as PyTorch, "
                    "TensorFlow, OpenCV, React, FastAPI, NumPy or Scikit-learn."
    )

    databases: list[str] = Field(
        default_factory=list,
        description="Databases explicitly mentioned, such as PostgreSQL, MySQL, MongoDB or Redis."
    )

    cloud_devops: list[str] = Field(
        default_factory=list,
        description="Cloud, containerization, deployment and DevOps technologies such as "
                    "AWS, Azure, GCP, Docker, Kubernetes or CI/CD."
    )

    tools_platforms: list[str] = Field(
        default_factory=list,
        description="Technical tools and platforms such as Git, Linux, ROS2, Tableau or Power BI."
    )

    softwares_used: list[str] = Field(
        default_factory=list,
        description="Standalone engineering, CAD, CAE, design or technical software such as "
                    "AutoCAD, Fusion 360, SolidWorks, ANSYS, Arduino IDE or Tinkercad."
    )

    other_technical_skills: list[str] = Field(
        default_factory=list,
        description="Other explicitly mentioned technical skills which do not clearly fit "
                    "into another technical skill category."
    )


class Profile(BaseModel):
    name: str | None = Field(
        default=None,
        description="Candidate's full name."
    )

    email: str | None = Field(
        default=None,
        description="Email address, null if absent."
    )

    phone: str | None = Field(
        default=None,
        description="Phone number, null if absent."
    )

    location: str | None = Field(
        default=None,
        description="City or region, null if absent."
    )

    relevant_courses: list[str] = Field(
        default_factory=list,
        description="Academic, university or online courses explicitly mentioned in the resume. "
                    "Do not put certifications here."
    )

    certifications: list[str] = Field(
        default_factory=list,
        description="Professional, technical or online certifications explicitly mentioned "
                    "anywhere in the resume."
    )

    experience: list[Experience] = Field(
        default_factory=list,
        description="All jobs and internships mentioned in the resume."
    )

    education: list[Education] = Field(
        default_factory=list,
        description="Every qualification mentioned including degrees, diplomas, doctorates "
                    "and school qualifications. Search the whole resume."
    )

    projects: list[Project] = Field(
        default_factory=list,
        description="All personal, academic, team and organizational projects."
    )

    technical_skills: TechnicalSkills = Field(
        default_factory=TechnicalSkills,
        description="Technical skills explicitly mentioned anywhere in the resume, categorized "
                    "appropriately. Search Skills, Technical Skills, Tools, Technologies, "
                    "Languages, Frameworks, project descriptions and experience descriptions. "
                    "Do not infer skills that are not explicitly present."
    )

    competitions: list[str] = Field(
        default_factory=list,
        description="Competitive exams, olympiads, hackathons, contests and competitions "
                    "together with any rank, score or placement stated. Copy each entry word for word."
    )

    positions_of_responsibility: list[Position] = Field(
        default_factory=list,
        description="Every role, membership or volunteering entry in a club, student team, "
                    "society, committee, chapter, cell, council, contingent, fest, NSS, NGO "
                    "or similar body. Resumes head this section in many different ways - "
                    "Positions of Responsibility, Leadership, Clubs, Teams, Societies, "
                    "Activities, Extra-Curricular, Volunteering, Community - so collect these "
                    "entries wherever they appear rather than only under one heading. "
                    "Being a member or a volunteer counts even when no title is given. "
                    "A paid job or internship is NOT one of these, it belongs in experience."
    )


# No local model on this chain. This step needs valid structured output against
# a deeply nested schema, and that already produced bracket-mismatch failures on
# a 70B model. A small local model would emit invalid JSON and lose the resume
# entirely, which is worse than failing loudly.
# Cerebras caps free-tier context at 8192 tokens and this call is ~6800, so a
# long resume overflows - that raises APIStatusError and the next model picks it
# up. Schema-validation failures are included too: if one model emits JSON that
# does not fit the schema, another one genuinely might.
PROFILE_FALLBACK_ERRORS = (
    RateLimitError,
    APIStatusError,
    APIConnectionError,
    # ChatCerebras subclasses BaseChatOpenAI, so it raises openai.* errors, and
    # openai.APIStatusError is NOT a subclass of groq.APIStatusError. Without
    # these three, a Cerebras 404 or 429 escapes the chain and kills the run.
    OpenAIRateLimitError,
    OpenAIAPIStatusError,
    OpenAIAPIConnectionError,
    GoogleAPIError,          # google, quota / 4xx / 5xx
    ContextOverflowError,    # resume longer than the model's context window
    ValidationError,         # pydantic, this model's JSON did not fit the schema
    OutputParserException,   # langchain could not read the reply at all
)

def trimmed_schema():
    """Profile's schema with the parts that teach the model nothing removed.

    Pydantic emits a "title" for every field ("Job Title", "Company", ...) and a
    "default" for every optional one. Both are pure repetition of the key name,
    and the whole schema is re-sent on every resume - so they cost ~284 tokens
    per resume for nothing. Descriptions are left completely alone: those carry
    the extraction rules and cutting them would cost accuracy.
    """
    schema = Profile.model_json_schema()

    def strip(node):
        if isinstance(node, dict):
            node.pop("title", None)
            node.pop("default", None)
            for value in node.values():
                strip(value)
        elif isinstance(node, list):
            for value in node:
                strip(value)
        return node

    return strip(schema)


structured_models = [m.with_structured_output(Profile) for m in MODEL_CHAIN]

structured_model = structured_models[0]

if len(structured_models) > 1:
    structured_model = structured_models[0].with_fallbacks(
        structured_models[1:],
        exceptions_to_handle=PROFILE_FALLBACK_ERRORS
    )


PROMPT = """
You are extracting structured information from the markdown text of a resume.

Rules:

- Extract only information explicitly present in the resume.
- Never invent or infer facts.
- If an optional value is absent, use null.
- If a list has no entries, return an empty list.
- Copy experience, project and leadership bullets word for word.
- Do not summarise or reword bullet points.
- Resumes often carry footnote markers such as *, dagger symbols or superscript
  numbers (*, +, 1, 2, 8, 10). Strip these from every value you extract.
  "8.36*" becomes "8.36", "JEE2 Advanced" becomes "JEE Advanced",
  "NSS10" becomes "NSS", "iBot Club, CFI8" becomes "iBot Club, CFI".
- Only strip a marker. A number that is genuinely part of the value stays, so
  "ROS 2", "Class XII" and "Python 3" keep their numbers.
- Convert dates to YYYY-MM where the month is available.
- If only a year is available, use YYYY.
- "Mar 2022" becomes "2022-03".
- "03/2022" becomes "2022-03".
- Present, ongoing or currently working should produce null as the end date.
- Categorize technical skills into the most appropriate TechnicalSkills field.
- Some resumes show how good the candidate is at a skill with a picture instead of
  words: a part-filled bar, a row of dots or stars where some are filled and some
  are empty, a percentage ring, or a slider. Read how full it is and write that
  level in brackets after the skill name, using the words beginner, intermediate,
  advanced or expert. Nearly full is expert, about three quarters is advanced,
  about half is intermediate, a quarter or less is beginner.
- "ETAP" with 4 of 5 dots filled becomes "ETAP (advanced, 4/5)".
- "Java" with a bar filled about a quarter becomes "Java (beginner, 1/4 bar)".
- If a skill has no such picture next to it, write the skill name on its own.
  Never invent a level that the resume does not show.
- A technology explicitly mentioned inside an experience or project bullet should also
  be included in that experience or project's technologies list.
- Do not infer broad domains. For example, YOLO may be extracted as a technology,
  but do not invent "Computer Vision" unless the resume itself states it.
- Keep certifications separate from relevant courses.
- The resume may contain unusual spacing or table formatting. Interpret the layout carefully.

Resume:
"""


def build_json_from_resume(markdown_text):
    response = structured_model.invoke(
        PROMPT + "\n\n" + markdown_text
    )

    with open("json_response.json", "w", encoding="utf8") as f:
        f.write(response.model_dump_json(indent=2))

    return response.model_dump()