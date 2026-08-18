"""Bias check over the top of the ranking.

This runs after the ranking is written and changes nothing about it. It looks
at who came out near the top and asks whether one institution, one department
or one band of academic scores is winning far more often than its share of the
applicant pool.

The band it examines is deliberately wider than the list of people who get an
interview email. Three invitations out of a large pile is far too few for a
selection ratio to mean anything - one candidate either way moves a group by
33 points - so the check is pointed at the top N of the ranking instead, with
N set by the caller. Invitations are unaffected either way.

It is a flag for a human, not a decision. A high selection ratio can be
completely legitimate - if only one person applied from a given college and
they were the strongest candidate, that is not bias. Everything here is
reported with the counts behind it so the reader can judge that themselves.
"""

import re

from build_evidence import write_block, write_field, REPORT_WIDTH


# A group has to be this big before a selection ratio means anything. With
# seven resumes almost every institution appears once, and one shortlisted
# candidate out of one applicant is a 100% selection ratio that says nothing.
# Groups under this size are still counted and printed, just not flagged.
MIN_GROUP = 3

# Above this, a candidate counts as "high academic score" for the score check.
# On a 10-point scale, after normalise_cgpa has put percentages on that scale.
CG_THRESHOLD = 9.0

# Selection ratio, as a percentage, above which a group gets flagged.
BIAS_LIMITS = {
    "institution": 70,
    "department": 80,
    "cg": 80
}


def normalise_cgpa(raw):
    """Academic score on a 10-point scale, or None if it cannot be read.

    The extractor copies whatever the resume printed, so this field arrives as
    a string and on two different scales - "7.96" is a CGPA, "78.5" and "91.4"
    are percentages. Comparing those directly would put every percentage above
    a threshold of 9 and the check would flag nothing but noise.

    Anything over 10 is treated as a percentage and divided by 10. Conversion
    factors genuinely vary by board and university, so this is an
    approximation. It is only used to bucket candidates for the grouping
    check, never to score anyone.
    """
    if raw is None:
        return None

    match = re.search(r"\d+(?:\.\d+)?", str(raw))

    if match is None:
        return None

    value = float(match.group())

    if value > 10:
        value = value / 10

    return value


def canonical(text):
    """A grouping key that ignores how the resume happened to punctuate a name.

    Institutions arrive exactly as each candidate typed them, so "Indian
    Institute of Technology Madras" and "Indian Institute of Technology,
    Madras" are the same place written two ways. Grouping on the raw string
    splits four applicants into two pairs, both of which then fall under
    MIN_GROUP and get dismissed as too small to judge - which is the check
    quietly doing nothing.

    Case, punctuation and repeated spaces are dropped. Nothing cleverer: a
    real alias list ("IIT Madras" against the full name) is a bigger job and
    this does not attempt it.
    """
    return " ".join(re.sub(r"[^\w\s]", " ", str(text)).lower().split())


def create_bias_data(ranked, band_size, invited_size=None):
    """One row per scored candidate: their group memberships and their outcome.

    `ranked` is what write_ranking returns - every resume that produced a
    score, sorted best first. Resumes that could not be read are already out of
    it, which is correct here: they were never eligible for the shortlist, so
    counting them would understate every selection ratio.

    `band_size` is what counts as selected for this check. `invited_size` is
    only recorded so the report can mark who actually gets an email; it takes
    no part in any ratio.

    Only the first education entry is used. It is the most recent qualification
    on the resume, and the ones below it are school records whose "field" is a
    board name rather than a department.
    """
    bias_data = {}

    for i, result in enumerate(ranked):

        candidate_id = f"candidate_{i + 1}"

        profile = result.get("profile") or {}
        education = profile.get("education") or []

        if len(education) > 0:
            institution = education[0].get("institution")
            department = education[0].get("field")
            cg = normalise_cgpa(education[0].get("cgpa"))
        else:
            institution = None
            department = None
            cg = None

        experience = len(profile.get("experience") or [])

        bias_data[candidate_id] = {
            "name": result.get("name"),
            "file": result.get("file"),
            "institution": institution,
            "department": department,
            "cg": cg,
            "experience": experience,
            "rank": i + 1,
            "shortlisted": i < band_size,
            "invited": invited_size is not None and i < invited_size
        }

    return bias_data


def feature_bias_check(bias_data, feature, bias_threshold, min_group=MIN_GROUP):
    """Selection ratio per value of one feature, and which values look skewed.

    Candidates whose value is missing are skipped rather than pooled into a
    "None" group, because "the extractor did not find an institution" is not
    an institution.
    """
    feature_counts = {}

    for candidate in bias_data.values():

        value = candidate.get(feature)

        if value is None:
            continue

        key = canonical(value)

        if key not in feature_counts:
            feature_counts[key] = {
                # the first spelling seen, so the report reads like a resume
                # rather than like a lookup key
                "label": value,
                "total": 0,
                "shortlisted": 0
            }

        feature_counts[key]["total"] += 1

        if candidate["shortlisted"]:
            feature_counts[key]["shortlisted"] += 1

    biased_values = []
    too_small = []

    for value, counts in feature_counts.items():

        total = counts["total"]
        shortlisted = counts["shortlisted"]

        ratio = (shortlisted / total) * 100
        counts["selection_ratio"] = round(ratio, 2)

        if total < min_group:
            too_small.append(counts["label"])
            continue

        if ratio > bias_threshold:

            biased_values.append({
                "value": counts["label"],
                "total": total,
                "shortlisted": shortlisted,
                "selection_ratio": round(ratio, 2),
                "bias": True
            })

    return {
        "feature": feature,
        "threshold": bias_threshold,
        "min_group": min_group,
        "feature_counts": feature_counts,
        "too_small_to_judge": too_small,
        "bias_found": len(biased_values) > 0,
        "biased_values": biased_values
    }


def cg_bias_check(bias_data, cg_threshold, bias_threshold):
    """How much of the shortlist comes from the high academic score band.

    This one is not grouped. It asks a single question: of everyone above the
    score threshold, what share got shortlisted? A very high number means the
    ranking is tracking academic score more than anything else in the resume.
    """
    total = 0
    shortlisted = 0

    for candidate in bias_data.values():

        cg = candidate.get("cg")

        if cg is None:
            continue

        if cg > cg_threshold:

            total += 1

            if candidate["shortlisted"]:
                shortlisted += 1

    if total == 0:
        ratio = 0
    else:
        ratio = (shortlisted / total) * 100

    return {
        "feature": "cg",
        "cg_threshold": cg_threshold,
        "bias_threshold": bias_threshold,
        "total_above_threshold": total,
        "shortlisted_above_threshold": shortlisted,
        "selection_ratio": round(ratio, 2),
        "bias_found": total >= MIN_GROUP and ratio > bias_threshold
    }


def write_bias_report(bias_data, checks, band_size, invited_size, output_path):
    """The whole check as one readable file, in the same shape as the others."""
    shortlisted = [c for c in bias_data.values() if c["shortlisted"]]

    with open(output_path, "w", encoding="utf8") as f:

        f.write("=" * REPORT_WIDTH + "\n")
        f.write("  BIAS CHECK ON THE TOP OF THE RANKING\n")
        f.write("=" * REPORT_WIDTH + "\n")

        write_field(f, "Pool", f"{len(bias_data)} scored candidates")
        write_field(f, "Band checked", f"top {band_size} ({len(shortlisted)} candidates)")

        if invited_size is not None:
            write_field(
                f, "Invitations",
                f"top {invited_size} - emailed, and not what this check measures"
            )

        write_field(f, "Min group", f"{MIN_GROUP} (smaller groups are counted, not flagged)")

        f.write("=" * REPORT_WIDTH + "\n\n")

        f.write("  CANDIDATES\n")
        f.write("  " + "-" * (REPORT_WIDTH - 2) + "\n")
        f.write(
            f"  {'#':>3}  {'IN BAND':<8} {'CANDIDATE':<22} {'CGPA':>6}  {'EXP':>3}  "
            f"{'DEPARTMENT':<24} INSTITUTION\n"
        )

        for candidate in bias_data.values():

            if candidate["invited"]:
                mark = "yes/mail"
            elif candidate["shortlisted"]:
                mark = "yes"
            else:
                mark = "-"

            cg = "-" if candidate["cg"] is None else f"{candidate['cg']:.2f}"

            f.write(
                f"  {candidate['rank']:>3}  {mark:<8} "
                f"{str(candidate['name'])[:22]:<22} {cg:>6}  "
                f"{candidate['experience']:>3}  "
                f"{str(candidate['department'])[:24]:<24} "
                f"{str(candidate['institution'])[:28]}\n"
            )

        f.write("\n")

        for check in checks:

            if check["feature"] == "cg":

                f.write(f"  ACADEMIC SCORE ABOVE {check['cg_threshold']}\n")
                f.write("  " + "-" * (REPORT_WIDTH - 2) + "\n")

                write_field(f, "Above bar", f"{check['total_above_threshold']} candidates")
                write_field(f, "Shortlisted", f"{check['shortlisted_above_threshold']}")
                write_field(f, "Ratio", f"{check['selection_ratio']}%  (flag over {check['bias_threshold']}%)")
                write_field(f, "Verdict", "FLAGGED" if check["bias_found"] else "nothing to flag")

                f.write("\n")
                continue

            f.write(f"  {check['feature'].upper()}\n")
            f.write("  " + "-" * (REPORT_WIDTH - 2) + "\n")
            f.write(
                f"  {'SHORT':>5} / {'TOTAL':<5}  {'RATIO':>7}   VALUE\n"
            )

            ordered = sorted(
                check["feature_counts"].items(),
                key=lambda item: item[1]["selection_ratio"],
                reverse=True
            )

            for value, counts in ordered:

                note = ""
                if counts["total"] < check["min_group"]:
                    note = "  (group too small to judge)"

                f.write(
                    f"  {counts['shortlisted']:>5} / {counts['total']:<5} "
                    f"{counts['selection_ratio']:>6.1f}%   "
                    f"{str(counts['label'])[:50]}{note}\n"
                )

            f.write("\n")

            if check["bias_found"]:
                write_block(
                    f,
                    f"FLAGGED: {len(check['biased_values'])} value(s) above "
                    f"{check['threshold']}% selection with at least "
                    f"{check['min_group']} applicants. Worth a human look.",
                    "  "
                )
            else:
                write_block(f, "Nothing to flag on this feature.", "  ")

            f.write("\n")

        f.write("  " + "-" * (REPORT_WIDTH - 2) + "\n")
        write_block(
            f,
            "This report does not change any score or any ranking. A high "
            "selection ratio is a reason to look, not evidence of unfairness "
            "on its own.",
            "  "
        )
        f.write("\n")

    return shortlisted


def run_bias_check(ranked, band_size, output_path="bias_report.txt",
                   invited_size=None):
    """Build the data, run all three checks, write the report, return the result.

    Returns None when there is nothing worth checking. That is either fewer
    than two scored candidates, or a band that already covers everyone who was
    scored - if nobody was left out, every group sits at a 100% selection ratio
    and the report would flag the whole field. Better to say so than to print a
    page of false positives.
    """
    if len(ranked) < 2:
        print("bias check skipped: fewer than two scored candidates")
        return None

    if band_size >= len(ranked):
        print(
            f"bias check skipped: the top-{band_size} band already covers all "
            f"{len(ranked)} scored candidates, so there is nobody to compare "
            f"them against"
        )
        return None

    bias_data = create_bias_data(ranked, band_size, invited_size)

    checks = [
        feature_bias_check(bias_data, "institution", BIAS_LIMITS["institution"]),
        feature_bias_check(bias_data, "department", BIAS_LIMITS["department"]),
        cg_bias_check(bias_data, CG_THRESHOLD, BIAS_LIMITS["cg"])
    ]

    write_bias_report(bias_data, checks, band_size, invited_size, output_path)

    flagged = [check["feature"] for check in checks if check["bias_found"]]

    if len(flagged) > 0:
        print(f"bias check: FLAGGED on {', '.join(flagged)} - see {output_path}")
    else:
        print(f"bias check: nothing flagged - see {output_path}")

    return {
        "bias_data": bias_data,
        "checks": checks,
        "flagged": flagged
    }
