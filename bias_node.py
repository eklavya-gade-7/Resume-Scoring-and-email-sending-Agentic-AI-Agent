def create_bias_data(candidates, shortlisted):

    bias_data = {}

    for i, candidate in enumerate(candidates):

        # Candidate name/id
        candidate_id = f"candidate_{i+1}"

        # Education
        education = candidate.education

        if education:
            institution = education[0].institution
            department = education[0].field
            cg = education[0].cgpa
        else:
            institution = None
            department = None
            cg = None
        # Experience
        
        experience = len(candidate.experience)

        # Shortlisted or not
    
        is_shortlisted = shortlisted[candidate_id]
        # Final dictionary
        # -----------------------------
        bias_data[candidate_id] = {
            "institution": institution,
            "department": department,
            "cg": cg,
            "experience": experience,
            "shortlisted": is_shortlisted
        }

    return bias_data

def feature_bias_check(bias_data, feature, bias_threshold):

    feature_counts = {}

    # Count candidates feature-wise
 
    for candidate in bias_data.values():

        value = candidate.get(feature)

        if value is None:
            continue

        if value not in feature_counts:
            feature_counts[value] = {
                "total": 0,
                "shortlisted": 0
            }

        feature_counts[value]["total"] += 1

        if candidate["shortlisted"]:
            feature_counts[value]["shortlisted"] += 1

    # Check bias
    # -----------------------------------------

    biased_values = []

    for value, counts in feature_counts.items():

        total = counts["total"]
        shortlisted = counts["shortlisted"]

        ratio = (shortlisted / total) * 100

        if ratio > bias_threshold:

            biased_values.append({
                "value": value,
                "total": total,
                "shortlisted": shortlisted,
                "selection_ratio": round(ratio, 2),
                "bias": True
            })


    return {
        "feature": feature,
        "threshold": bias_threshold,
        "feature_counts": feature_counts,
        "bias_found": len(biased_values) > 0,
        "biased_values": biased_values
    }

def cg_bias_check(bias_data, cg_threshold, bias_threshold):

    total = 0
    shortlisted = 0

    for candidate in bias_data.values():

        cg = candidate.get("cg")

        if cg is None:
            continue

        try:
            cg = float(cg)
        except (ValueError, TypeError):
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
        "bias_found": ratio > bias_threshold
    }

BIAS_LIMITS = {
    "institution": 70,
    "department": 80,
    "cg":80
}
CG_THRESHOLD = 9

institution_result = feature_bias_check(
    bias_data,
    "institution",
    BIAS_LIMITS["institution"]
)

department_result = feature_bias_check(
    bias_data,
    "department",
    BIAS_LIMITS["department"]
)
cg_result = cg_bias_check(
    bias_data,
    CG_THRESHOLD,
    BIAS_LIMITS["cg"]
)



