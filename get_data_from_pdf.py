import os
import base64

import pymupdf
import pymupdf4llm as ppdf
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
from groq import RateLimitError, APIStatusError, APIConnectionError
from pydantic import ValidationError

from build_profile import Profile, PROMPT as EXTRACTION_RULES, build_json_from_resume

# from openai import OpenAI

load_dotenv()              # reads the API keys from a .env file in this folder

cwd = os.getcwd()

# ---- OpenAI version (kept for reference, not in use) ----
# openai_api_key = os.getenv("OPENAI_API_KEY")
# client = OpenAI(api_key = openai_api_key)
# VLM_MODEL = "gpt-4o-mini"

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise SystemExit("GEMINI_API_KEY not found. Add it to your .env file.")

client = genai.Client(api_key=gemini_api_key)

VLM_MODEL = "gemini-3.5-flash"

# Gemini free-tier limits are counted per model, so a second Gemini model is a
# genuinely separate bucket rather than the same quota under another name.
VLM_MODEL_LITE = "gemini-flash-lite-latest"


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
    """Second tier. Different provider, so a completely separate daily budget."""
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


def callOCR_paddle(path):
    """Third tier. Local, never rate limited, but text only.

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


# Tried in order, first one that returns text wins.
VLM_CHAIN = [
    ("gemini flash", callVLM),
    ("gemini flash lite", callVLM_gemini_lite),
    ("groq llama-4-scout", callVLM_groq),
    ("paddleocr (local, no layout)", callOCR_paddle),
]


# Only these mean "this reader is unavailable, try the next one". Anything else
# is a real bug and should surface instead of quietly sliding down the chain.
READER_ERRORS = (
    ClientError,             # google, 429 quota and 4xx
    ServerError,             # google, 5xx
    RateLimitError,          # groq
    APIStatusError,          # groq
    APIConnectionError,      # groq
    RuntimeError,            # our own guards, e.g. too many pages for Scout
    ImportError,             # paddleocr missing or broken install
    OSError,                 # model download / disk failure
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
    md = clean_tags(ppdf.to_markdown(path, use_ocr=False))
    raw_text = get_raw_text(path)

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

