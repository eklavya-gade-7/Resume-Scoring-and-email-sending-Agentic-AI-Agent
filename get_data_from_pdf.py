import os
import base64

import pymupdf
import pymupdf4llm as ppdf
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
from groq import RateLimitError, APIStatusError, APIConnectionError
# Aliased, and it matters. openai.APIStatusError and groq.APIStatusError are two
# unrelated classes that happen to share a name - importing both unaliased means
# the second silently replaces the first, and one provider's errors stop being
# caught at all. Same trap as the one documented in generate_score.py.
from openai import (
    OpenAI,
    RateLimitError as OpenAIRateLimitError,
    APIStatusError as OpenAIAPIStatusError,
    APIConnectionError as OpenAIAPIConnectionError,
)
from pydantic import ValidationError

from build_profile import Profile, PROMPT as EXTRACTION_RULES, build_json_from_resume

load_dotenv()              # reads the API keys from a .env file in this folder

cwd = os.getcwd()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise SystemExit("GEMINI_API_KEY not found. Add it to your .env file.")

client = genai.Client(api_key=gemini_api_key)

VLM_MODEL = "gemini-3.5-flash"

# Gemini free-tier limits are counted per model, so a second Gemini model is a
# genuinely separate bucket rather than the same quota under another name.
VLM_MODEL_LITE = "gemini-flash-lite-latest"

# Both of these speak the OpenAI wire format, so they need no new dependency -
# the openai client already installed just gets pointed somewhere else.
#
# Model names are the fragile part. This project has already lost two models to
# silent retirement (Groq's llama-4-scout and llama-3.3-70b), so check a name is
# still served before blaming the code:
#   OpenRouter  curl https://openrouter.ai/api/v1/models     (no key needed)
#   Mistral     curl -H "Authorization: Bearer $MISTRAL_API_KEY" \
#                    https://api.mistral.ai/v1/models
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"

# Mistral lists mistral-large-2512, mistral-medium-2508, mistral-small-2506 and
# the ministral-3b/8b/14b-2512 family as the ones that can see images. The small
# one is plenty for transcribing a page and is the kindest to a free tier.
# (pixtral-12b used to be the vision model here and is no longer on that list.)
MISTRAL_VLM_MODEL = "mistral-small-latest"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Picked off the live free list. Free models on OpenRouter come and go monthly,
# so this one is a constant rather than buried in the call.
OPENROUTER_VLM_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"

# Conservative ceiling shared by both. A resume is one or two pages, so this
# only ever trips on something that was not really a resume.
MAX_PAGES_PER_REQUEST = 8


def get_raw_text(path):
    raw_text = ""
    doc = pymupdf.open(path)
    for page in doc:
        raw_text = raw_text + page.get_text()
    return raw_text


def clean_tags(text):
    for tag in ["<u>", "</u>", "<mark>", "</mark>", "<sup>", "</sup>"]:
        text = text.replace(tag, "")
    text = text.replace("<br>", "\n")
    return text


def find_missing_words(raw_text, markdown_text):

    missing = []
    for word in raw_text.split():
        word = word.strip(".,()")
        if len(word) > 4 and word not in markdown_text:
            missing.append(word)
    return missing


PROMPT = """Transcribe this resume page into markdown.

Rules:
- Transcribe everything. Do not summarise, shorten or reword anything.
- Use ## for section headings (EXPERIENCE, EDUCATION, SKILLS and so on).
- Use ** for job titles and company names.
- Keep bullet points as bullets.
- Copy dates exactly as written.
- For two columns, finish one column completely before starting the other.
- If something is unreadable write [unclear]. Never guess.
Output only the markdown."""


def page_images(path):
    """Every page of the PDF as PNG bytes, about 2000px on the long edge."""
    images = []
    doc = pymupdf.open(path)
    for page in doc:
        long_side = max(page.rect.width, page.rect.height)
        dpi = int(72 * 2000 / long_side)
        images.append(page.get_pixmap(dpi=dpi).tobytes("png"))
    return images


def callVLM(path):
    parts = [PROMPT]
    for png in page_images(path):
        parts.append(types.Part.from_bytes(data=png, mime_type="image/png"))

    reply = client.models.generate_content(
        model=VLM_MODEL,
        contents=parts,
    )
    return reply.text


class ExtractionFailed(Exception):
    """Nothing could read this file. The resume must not be scored.

    An empty markdown produces an empty profile, which scores 0 - and a 0 looks
    exactly like a genuinely weak candidate. Failing loudly keeps a broken scan
    out of the ranking instead of quietly putting it last.
    """


def callVLM_gemini_lite(path):
    """Second tier. Same provider, different model, so a fresh per-model quota."""
    parts = [PROMPT]
    for png in page_images(path):
        parts.append(types.Part.from_bytes(data=png, mime_type="image/png"))

    reply = client.models.generate_content(
        model=VLM_MODEL_LITE,
        contents=parts,
    )
    return reply.text


def callVLM_groq(path):
    """Not wired into VLM_CHAIN any more. Kept for when Groq serves vision again.

    Groq retired llama-4-scout and now serves no image model at all - asking for
    it returns 404 model_not_found. The tier could never succeed, so it was
    removed from the chain rather than left there failing on every scan. Point
    the model name at whatever Groq offers next and put it back in the list.
    """
    from groq import Groq

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    images = page_images(path)

    # Scout accepts at most 5 images per request
    if len(images) > 5:
        raise RuntimeError(f"{len(images)} pages, Scout accepts 5")

    content = [{"type": "text", "text": PROMPT}]
    for png in images:
        b64 = base64.b64encode(png).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })

    reply = Groq(api_key=groq_api_key).chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": content}],
        temperature=0
    )
    return reply.choices[0].message.content


def data_uris(path):
    """Every page as a base64 data URI, with a sane page-count guard.

    Both OpenAI-compatible tiers below want images inline rather than as a
    hosted URL, and neither of them will accept a 40-page document.
    """
    images = page_images(path)

    if len(images) > MAX_PAGES_PER_REQUEST:
        raise RuntimeError(
            f"{len(images)} pages, this tier accepts {MAX_PAGES_PER_REQUEST}"
        )

    return [
        "data:image/png;base64," + base64.b64encode(png).decode()
        for png in images
    ]


def callVLM_mistral(path):
    """Third tier. Different company, so a completely separate budget.

    Mistral's free Experiment tier is a monthly token allowance rather than a
    per-day one, which makes it a genuinely different kind of budget from
    Gemini's per-minute request cap - the failure that pushed us this far down
    the chain in the first place.
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY not set")

    # Mistral takes image_url as a plain string, NOT the nested {"url": ...}
    # object OpenAI and OpenRouter use. Same field name, different shape. If
    # this tier starts returning 422, that is the first thing to check.
    content = [{"type": "text", "text": PROMPT}]
    for uri in data_uris(path):
        content.append({"type": "image_url", "image_url": uri})

    reply = OpenAI(api_key=api_key, base_url=MISTRAL_BASE_URL).chat.completions.create(
        model=MISTRAL_VLM_MODEL,
        messages=[{"role": "user", "content": content}],
        temperature=0
    )
    return reply.choices[0].message.content


def callVLM_openrouter(path):
    """Fourth tier. One key, and a pool of free vision models behind it.

    Worth knowing before leaning on this: free models on OpenRouter allow 20
    requests a minute but only 50 a day until the account has bought $10 of
    credit at some point, after which it is 1000 a day. Fine as a backstop for
    the occasional scan, not something to run a whole batch through.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    content = [{"type": "text", "text": PROMPT}]
    for uri in data_uris(path):
        content.append({"type": "image_url", "image_url": {"url": uri}})

    reply = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL).chat.completions.create(
        model=OPENROUTER_VLM_MODEL,
        messages=[{"role": "user", "content": content}],
        temperature=0
    )
    return reply.choices[0].message.content


def callOCR_paddle(path):
    """Last tier. Local, never rate limited, but text only.

    There is no layout here. The education table becomes loose lines, so the
    CGPA may no longer sit next to the degree it belongs to. Expect a weaker
    profile from this tier, not an equivalent one.
    """
    import tempfile
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(lang="en", use_textline_orientation=True)

    lines = []

    with tempfile.TemporaryDirectory() as folder:

        for number, png in enumerate(page_images(path)):

            image_path = os.path.join(folder, f"page_{number}.png")

            with open(image_path, "wb") as f:
                f.write(png)

            for page_result in ocr.predict(image_path):

                # paddleocr 3.x returns dicts keyed by rec_texts
                if isinstance(page_result, dict) and "rec_texts" in page_result:
                    lines.extend(page_result["rec_texts"])

                # older releases return [box, (text, confidence)] per line
                else:
                    for entry in page_result or []:
                        lines.append(entry[1][0])

    return "\n".join(lines)


# The same extraction rules build_profile uses, so the JSON comes out identical
# whichever route produced it. Only the layout rules are added, because those
# are the ones that only make sense when the model can actually see the page.
JSON_PROMPT = (
    "You are extracting structured information directly from the page images "
    "of a resume.\n\nRules:\n"
    + EXTRACTION_RULES.split("Rules:", 1)[1].replace("Resume:", "").rstrip()
    + "\n- You can see the page layout, so use it. A value sitting in a table"
      "\n  column belongs to the row it is in - a CGPA in the CGPA column"
      "\n  belongs to the degree on that same row."
    + "\n- For a two-column layout, read one column completely before the other."
    + "\n- If text is unreadable, leave that field null. Never guess.\n"
)


def callVLM_json(path, model_name):
    """Read the page images and return a Profile, skipping markdown entirely.

    Markdown flattens the education table, and a text-only model then has to
    re-infer which number belongs to which row - which is exactly how the CGPA
    went missing. A model looking at the page sees a table as a table.

    Gemini uses constrained decoding for response_schema, so unlike tool-call
    validation it cannot physically emit JSON that breaks the schema.
    """
    parts = [JSON_PROMPT]
    for png in page_images(path):
        parts.append(types.Part.from_bytes(data=png, mime_type="image/png"))

    reply = client.models.generate_content(
        model=model_name,
        contents=parts,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Profile,
        ),
    )

    profile = reply.parsed

    if profile is None:

        # reply.text is None whenever the model produced no usable output at
        # all - blocked, truncated at the token limit, or no candidate returned.
        # Feeding that straight to model_validate_json reports "JSON input
        # should be string", which hides the actual reason.
        if not reply.text:

            candidates = reply.candidates or []
            finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None

            raise ValueError(
                f"model returned no JSON for {os.path.basename(path)} "
                f"(finish_reason={finish_reason}, "
                f"prompt_feedback={getattr(reply, 'prompt_feedback', None)}, "
                f"candidates={len(candidates)})"
            )

        profile = Profile.model_validate_json(reply.text)

    return profile


# Only the Gemini tiers. Scout's support for images plus a response schema is
# unverified, and PaddleOCR returns loose text - both keep the markdown route.
JSON_CHAIN = [
    ("gemini flash (direct JSON)", lambda path: callVLM_json(path, VLM_MODEL)),
    ("gemini flash lite (direct JSON)", lambda path: callVLM_json(path, VLM_MODEL_LITE)),
]


def looks_empty(profile):
    """True when a profile came back technically valid but with nothing in it."""
    return not (
        profile.name
        or profile.experience
        or profile.education
        or profile.projects
    )


def read_profile_with_fallbacks(path):
    """Try each Gemini tier for direct JSON. Returns a Profile, or None."""
    for name, reader in JSON_CHAIN:

        try:
            profile = reader(path)

            if profile is not None and not looks_empty(profile):
                print(f"  profile built directly by {name}")
                return profile

            print(f"  {name} returned an empty profile, trying next")

        except JSON_READER_ERRORS as error:
            print(f"  {name} failed ({type(error).__name__}: {error}), trying next")

    return None


def build_vlm_chain():
    """Ordered readers for page images -> markdown, best first.

    Tiers whose key is missing are left out rather than included and allowed to
    fail. Both amount to the same markdown in the end, but a chain built from
    the keys that actually exist prints one honest line per resume instead of a
    guaranteed failure for every provider the user never signed up for.

    Groq used to sit between the Gemini tiers and PaddleOCR. It no longer serves
    any image model, so that tier was a guaranteed 404 on every scan and is
    gone - see callVLM_groq.
    """
    chain = [
        ("gemini flash", callVLM),
        ("gemini flash lite", callVLM_gemini_lite),
    ]

    if os.getenv("MISTRAL_API_KEY"):
        chain.append((f"mistral {MISTRAL_VLM_MODEL}", callVLM_mistral))

    if os.getenv("OPENROUTER_API_KEY"):
        chain.append((f"openrouter {OPENROUTER_VLM_MODEL}", callVLM_openrouter))

    # Always last. It is the only tier that cannot be rate limited, but it reads
    # text without layout, so everything above it gets first refusal.
    chain.append(("paddleocr (local, no layout)", callOCR_paddle))

    return chain


# Tried in order, first one that returns text wins.
VLM_CHAIN = build_vlm_chain()


# Only these mean "this reader is unavailable, try the next one". Anything else
# is a real bug and should surface instead of quietly sliding down the chain.
READER_ERRORS = (
    ClientError,                 # google, 429 quota and 4xx
    ServerError,                 # google, 5xx
    RateLimitError,              # groq
    APIStatusError,              # groq
    APIConnectionError,          # groq
    OpenAIRateLimitError,        # mistral and openrouter, out of quota
    OpenAIAPIStatusError,        # mistral and openrouter, 4xx and 5xx
    OpenAIAPIConnectionError,    # mistral and openrouter, network
    RuntimeError,                # our own guards: missing key, too many pages
    ImportError,                 # paddleocr missing or broken install
    OSError,                     # model download / disk failure
)

# The direct-JSON path can also fail on the schema itself: a model may return
# valid JSON that does not fit Profile, or no parseable JSON at all.
JSON_READER_ERRORS = READER_ERRORS + (ValidationError, ValueError)


def read_with_fallbacks(path):
    """Walk the chain until something returns text. Returns "" if all fail."""
    for name, reader in VLM_CHAIN:

        try:
            text = reader(path)

            if text and text.strip():
                print(f"  extracted with {name}")
                return text

            print(f"  {name} returned nothing, trying next")

        except READER_ERRORS as error:
            print(f"  {name} failed ({type(error).__name__}: {error}), trying next")

    return ""


# ---- OpenAI version (kept for reference, not in use) ----
# def callVLM(path):
#     content = [{"type": "text", "text": PROMPT}]
#     for png in page_images(path):
#         b64 = base64.b64encode(png).decode()
#         content.append({
#             "type": "image_url",
#             "image_url": {"url": f"data:image/png;base64,{b64}"},
#         })
#
#     reply = client.chat.completions.create(
#         model=VLM_MODEL,
#         messages=[{"role": "user", "content": content}],
#     )
#     return reply.choices[0].message.content


def extract(path):
    md = ppdf.to_markdown(path, use_ocr=False)
    md = clean_tags(md)
    raw_text = get_raw_text(path)

    if len(raw_text) < 150:
        print("VLM called - no text layer, this is a scan")
        vlm_md = read_with_fallbacks(path)

        # A scan has no text layer, so there is no md to fall back to - it is
        # empty. Nothing readable came out, so refuse to score this resume.
        if not vlm_md.strip():
            raise ExtractionFailed(
                f"{os.path.basename(path)} is a scan and every reader failed. "
                "Send it for manual review - do not score it."
            )

        print("VLM output cannot be checked - a scan has no reference text")
        return vlm_md

    missing = find_missing_words(raw_text, md)
    print(f"missing words = {len(missing)}")

    if len(missing) > 12:
        print(f"VLM called - pymupdf4llm lost {len(missing)} words")
        vlm_md = read_with_fallbacks(path)

        # Last tier, and only usable here. This PDF has a real text layer, so
        # the pymupdf4llm output is imperfect but far from useless - unlike the
        # scan branch above, where it would be empty.
        if not vlm_md.strip():
            print("every reader failed, keeping the pymupdf4llm output instead")
            with open("extracted_markdown4.md", "w", encoding="utf8") as f:
                f.write(md)
            return md

        vlm_missing = find_missing_words(raw_text, vlm_md)
        print(f"pymupdf4llm lost {len(missing)} words, VLM lost {len(vlm_missing)}")
        if vlm_missing:
            print(f"still missing after VLM: {vlm_missing[:15]}")
        with open("extracted_markdown4.md", "w", encoding="utf8") as f:
            f.write(vlm_md)
        print(f"done, {len(vlm_md)} chars written")
        return vlm_md

    with open("extracted_markdown4.md", "w", encoding="utf8") as f:
        f.write(md)
    print(f"done, {len(md)} chars written")
    return md


def get_profile(path):
    """Resume file -> Profile dict, identical in shape whichever route ran.

    Fast path: if the page needs an image model anyway, that model emits the
    JSON in one call and build_profile is skipped entirely.

    Slow path: a clean digital PDF never needed an image model, so its markdown
    goes through build_profile as before - and so does anything the direct-JSON
    attempt failed on.
    """
    # A file can carry a .pdf name and still not be a PDF - a truncated
    # download, an HTML error page saved with the wrong extension, or a
    # password-protected export. pymupdf raises FileDataError on all of those,
    # and it is not an ExtractionFailed, so it used to escape every handler and
    # end the batch. It is exactly the same class of problem as a reader giving
    # up, so it is reported the same way.
    try:
        md = clean_tags(ppdf.to_markdown(path, use_ocr=False))
        raw_text = get_raw_text(path)

    except pymupdf.FileDataError as error:
        raise ExtractionFailed(
            f"{os.path.basename(path)} could not be opened as a PDF ({error}). "
            "It may be corrupt, password protected, or not a PDF at all. "
            "Send it for manual review."
        ) from error

    is_scan = len(raw_text.strip()) < 150
    missing = [] if is_scan else find_missing_words(raw_text, md)

    if not is_scan and len(missing) <= 12:
        print(f"missing words = {len(missing)}, markdown is faithful, no image model needed")
        with open("extracted_markdown4.md", "w", encoding="utf8") as f:
            f.write(md)
        return build_json_from_resume(md)

    if is_scan:
        print("scan detected, going straight to JSON from the page images")
    else:
        print(f"pymupdf4llm lost {len(missing)} words, going straight to JSON from the page images")

    profile = read_profile_with_fallbacks(path)

    if profile is not None:
        # written exactly as build_json_from_resume writes it, so the file on
        # disk is the same whichever route produced it
        with open("json_response.json", "w", encoding="utf8") as f:
            f.write(profile.model_dump_json(indent=2))
        return profile.model_dump()

    print("  direct JSON failed on every Gemini tier, falling back to markdown")
    markdown_text = extract(path)
    return build_json_from_resume(markdown_text)


path = f"{cwd}/Image_60.pdf"

