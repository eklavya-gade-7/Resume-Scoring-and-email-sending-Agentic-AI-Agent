from datetime import datetime


def calculate_duration_months(start, end):

    if start is None:
        return None, None

    # A year sometimes arrives as the number 2022 rather than the string
    # "2022", and len() of an int is a TypeError. Everything below this line
    # measures the length of the text, so make it text first.
    start = str(start)

    if end is not None:
        end = str(end)

    # "2022-13" is seven characters and passes the length test, but strptime
    # rejects month 13. An unparseable date is not worth failing a resume over
    # - the duration is simply unknown, which is what None already means here.
    try:
        return months_between(start, end)
    except ValueError:
        return None, None


def months_between(start, end):

    if len(start) == 7:
        start_earliest = datetime.strptime(start, "%Y-%m")
        start_latest = start_earliest

    elif len(start) == 4:
        year = int(start)
        start_earliest = datetime(year, 1, 1)
        start_latest = datetime(year, 12, 1)

    else:
        return None, None

    if end is None:
        current = datetime.now()
        end_earliest = current
        end_latest = current

    elif len(end) == 7:
        end_earliest = datetime.strptime(end, "%Y-%m")
        end_latest = end_earliest

    elif len(end) == 4:
        year = int(end)
        end_earliest = datetime(year, 1, 1)
        end_latest = datetime(year, 12, 1)

    else:
        return None, None

    minimum_months = (
        (end_earliest.year - start_latest.year) * 12
        + (end_earliest.month - start_latest.month)
    )

    maximum_months = (
        (end_latest.year - start_earliest.year) * 12
        + (end_latest.month - start_earliest.month)
    )

    minimum_months = max(0, minimum_months)
    maximum_months = max(0, maximum_months)

    return minimum_months, maximum_months


def clean_text(text):
    if text is None:
        return None

    return text.replace("**", "").replace("__", "").strip()


def remove_duplicates(values):

    answer = []

    # `values or []` rather than plain `values`. A field the extractor filled
    # with an explicit null reaches here as None, and dict.get("x", []) does NOT
    # return the default for a key that exists with a null value - it returns
    # the null. Iterating that is a TypeError that kills the whole batch.
    for value in values or []:

        if value is None:
            continue

        # Extractors sometimes put a number where a string belongs, most often a
        # bare year. str() keeps the value instead of dropping it.
        if not isinstance(value, str):
            value = str(value)

        value = clean_text(value)

        if value != "" and value not in answer:
            answer.append(value)

    return answer


def entries(value):
    """The dict entries of a resume list field, whatever the extractor produced.

    education, experience, projects and the rest are meant to be lists of
    dicts, and usually are. But the model writing them is not bound to that:
    it can return null for a section the resume does not have, a single dict
    where the resume had exactly one entry, or a list of plain strings like
    ["IIT Madras"] when it decides to flatten the shape.

    Every one of those crashed the chunker rather than the one resume - a null
    on len(), a string on .get(). Normalising here keeps a slightly odd
    extraction scoreable instead of sending it to a human.
    """
    if value is None:
        return []

    if isinstance(value, dict):
        return [value]

    if not isinstance(value, list):
        return []

    return [item for item in value if isinstance(item, dict)]


def create_structural_chunks_from_resume(resume):

    chunks = []

    technical_skills = resume.get("technical_skills")

    # `or {}` alone is not enough. It catches null and {}, but a model that
    # returns a plain string here - "technical_skills": "Python, C++" - is
    # truthy, sails past it, and dies on .get() a line later.
    if not isinstance(technical_skills, dict):
        technical_skills = {}

    programming_languages = remove_duplicates(
        technical_skills.get("programming_languages", [])
    )

    if len(programming_languages) > 0:
        chunks.append({
            "chunk_id": "skills_programming_languages",
            "section": "technical_skills",
            "text": (
                "Candidate programming languages: "
                + ", ".join(programming_languages)
            ),
            "metadata": {
                "skill_category": "programming_languages",
                "skills": programming_languages
            }
        })


    frameworks_libraries = remove_duplicates(
        technical_skills.get("frameworks_libraries", [])
    )

    if len(frameworks_libraries) > 0:
        chunks.append({
            "chunk_id": "skills_frameworks_libraries",
            "section": "technical_skills",
            "text": (
                "Candidate frameworks and libraries: "
                + ", ".join(frameworks_libraries)
            ),
            "metadata": {
                "skill_category": "frameworks_libraries",
                "skills": frameworks_libraries
            }
        })


    databases = remove_duplicates(
        technical_skills.get("databases", [])
    )

    cloud_devops = remove_duplicates(
        technical_skills.get("cloud_devops", [])
    )

    databases_cloud = remove_duplicates(
        databases + cloud_devops
    )

    if len(databases_cloud) > 0:
        chunks.append({
            "chunk_id": "skills_databases_cloud_devops",
            "section": "technical_skills",
            "text": (
                "Candidate databases, cloud and DevOps technologies: "
                + ", ".join(databases_cloud)
            ),
            "metadata": {
                "skill_category": "databases_cloud_devops",
                "skills": databases_cloud
            }
        })


    tools_platforms = remove_duplicates(
        technical_skills.get("tools_platforms", [])
    )

    softwares_used = remove_duplicates(
        technical_skills.get("softwares_used", [])
    )

    other_skills = remove_duplicates(
        technical_skills.get("other_technical_skills", [])
    )

    tools_software = remove_duplicates(
        tools_platforms + softwares_used + other_skills
    )

    if len(tools_software) > 0:
        chunks.append({
            "chunk_id": "skills_tools_software",
            "section": "technical_skills",
            "text": (
                "Candidate technical tools, platforms and software: "
                + ", ".join(tools_software)
            ),
            "metadata": {
                "skill_category": "tools_software",
                "skills": tools_software
            }
        })


    relevant_courses = remove_duplicates(
        resume.get("relevant_courses", [])
    )

    if len(relevant_courses) > 0:
        chunks.append({
            "chunk_id": "relevant_courses_0",
            "section": "relevant_courses",
            "text": (
                "Candidate relevant coursework: "
                + ", ".join(relevant_courses)
            ),
            "metadata": {
                "relevant_courses": relevant_courses
            }
        })


    certifications = remove_duplicates(
        resume.get("certifications", [])
    )

    if len(certifications) > 0:
        chunks.append({
            "chunk_id": "certifications_0",
            "section": "certifications",
            "text": (
                "Candidate certifications: "
                + ", ".join(certifications)
            ),
            "metadata": {
                "certifications": certifications
            }
        })


    experiences = entries(resume.get("experience"))

    for i in range(len(experiences)):

        experience = experiences[i]

        company = experience.get("company", None)
        job_title = experience.get("job_title", None)
        start = experience.get("start", None)
        end = experience.get("end", None)

        minimum_months, maximum_months = calculate_duration_months(
            start,
            end
        )

        technologies = remove_duplicates(
            experience.get("technologies", [])
        )

        bullets = remove_duplicates(
            experience.get("bullets", [])
        )

        parts = ["Candidate professional experience."]

        if job_title is not None:
            parts.append(f"Role: {job_title}.")

        if company is not None:
            parts.append(f"Company: {company}.")

        if start is not None:
            end_text = end if end is not None else "Present"
            parts.append(f"Period: {start} to {end_text}.")

        if minimum_months is not None:

            if minimum_months == maximum_months:
                parts.append(
                    f"Duration in this role: {minimum_months} months."
                )

            else:
                parts.append(
                    f"Duration in this role is between "
                    f"{minimum_months} and {maximum_months} months."
                )

        if len(technologies) > 0:
            parts.append(
                "Technologies used: "
                + ", ".join(technologies)
                + "."
            )

        if len(bullets) > 0:
            parts.append(
                "Experience details:\n"
                + "\n".join(
                    [f"- {bullet}" for bullet in bullets]
                )
            )

        chunks.append({
            "chunk_id": f"experience_{i}",
            "section": "experience",
            "text": "\n".join(parts),
            "metadata": {
                "company": company,
                "job_title": job_title,
                "start": start,
                "end": end,
                "minimum_duration_months": minimum_months,
                "maximum_duration_months": maximum_months,
                "technologies": technologies
            }
        })


    education = entries(resume.get("education"))

    for i in range(len(education)):

        edu = education[i]

        institution = edu.get("institution", None)
        degree = edu.get("degree", None)
        field = edu.get("field", None)
        year = edu.get("year", None)
        cgpa = edu.get("cgpa", None)

        parts = ["Candidate education."]

        if degree is not None:
            parts.append(
                f"Degree or qualification: {degree}."
            )

        if field is not None:
            parts.append(
                f"Department or field of study: {field}."
            )

        if institution is not None:
            parts.append(
                f"Institution: {institution}."
            )

        if cgpa is not None:
            parts.append(
                f"CGPA, GPA, percentage or academic score: {cgpa}."
            )

        if year is not None:
            parts.append(
                f"Completion or graduation year: {year}."
            )

        chunks.append({
            "chunk_id": f"education_{i}",
            "section": "education",
            "text": "\n".join(parts),
            "metadata": {
                "institution": institution,
                "degree": degree,
                "field": field,
                "year": year,
                "cgpa": cgpa
            }
        })


    projects = entries(resume.get("projects"))

    for i in range(len(projects)):

        project = projects[i]

        project_name = project.get("name", None)

        technologies = remove_duplicates(
            project.get("technologies", [])
        )

        bullets = remove_duplicates(
            project.get("bullets", [])
        )

        parts = ["Candidate project."]

        if project_name is not None:
            parts.append(
                f"Project name: {project_name}."
            )

        if len(technologies) > 0:
            parts.append(
                "Technologies used: "
                + ", ".join(technologies)
                + "."
            )

        if len(bullets) > 0:
            parts.append(
                "Project details:\n"
                + "\n".join(
                    [f"- {bullet}" for bullet in bullets]
                )
            )

        chunks.append({
            "chunk_id": f"project_{i}",
            "section": "projects",
            "text": "\n".join(parts),
            "metadata": {
                "project_name": project_name,
                "technologies": technologies
            }
        })


    competitions = remove_duplicates(
        resume.get("competitions", [])
    )

    if len(competitions) > 0:
        chunks.append({
            "chunk_id": "competitions_0",
            "section": "competitions",
            "text": (
                "Candidate competitions, exams and achievements:\n"
                + "\n".join(
                    [f"- {competition}" for competition in competitions]
                )
            ),
            "metadata": {
                "competitions": competitions
            }
        })


    positions = entries(resume.get("positions_of_responsibility"))

    simple_positions = []

    for i in range(len(positions)):

        position = positions[i]

        role = position.get("role", None)
        organization = position.get("organization", None)

        bullets = remove_duplicates(
            position.get("bullets", [])
        )

        if len(bullets) == 0:

            text = ""

            if role is not None:
                text += role

            if organization is not None:

                if text != "":
                    text += " - "

                text += organization

            if text != "":
                simple_positions.append(text)

            continue

        parts = ["Candidate leadership or position of responsibility."]

        if role is not None:
            parts.append(
                f"Role: {role}."
            )

        if organization is not None:
            parts.append(
                f"Organization: {organization}."
            )

        parts.append(
            "Responsibilities and achievements:\n"
            + "\n".join(
                [f"- {bullet}" for bullet in bullets]
            )
        )

        chunks.append({
            "chunk_id": f"position_{i}",
            "section": "positions_of_responsibility",
            "text": "\n".join(parts),
            "metadata": {
                "role": role,
                "organization": organization
            }
        })


    if len(simple_positions) > 0:
        chunks.append({
            "chunk_id": "positions_short_0",
            "section": "positions_of_responsibility",
            "text": (
                "Candidate positions of responsibility:\n"
                + "\n".join(
                    [f"- {position}" for position in simple_positions]
                )
            ),
            "metadata": {
                "positions": simple_positions
            }
        })


    return chunks