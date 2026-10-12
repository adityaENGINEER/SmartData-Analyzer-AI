"""
Deterministic and Explainable Data Quality Scoring Engine for DataDoctor AI.
Generates an interpretable 0-100 score based on missingness, duplicates, structural consistency,
and data validity with granular component breakdowns.
"""

from typing import Dict, Any, List


class QualityScoreCalculator:
    """Calculates an explainable 0 to 100 Data Quality Score with transparent penalty weights."""

    WEIGHTS = {
        "completeness": 35.0,  # Missing values & empty columns/rows
        "uniqueness": 25.0,    # Duplicate records
        "consistency": 20.0,   # Constant columns, whitespace anomalies, text numbers
        "integrity": 20.0,     # Outliers, valid record structures
    }

    @classmethod
    def calculate_score(cls, profile_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates the quality score from profile metrics.
        Returns overall score, component scores, deductions, and top impacting issues.
        """
        total_rows = profile_result.get("total_rows", 0)
        total_cols = profile_result.get("total_cols", 0)

        if total_rows == 0 or total_cols == 0:
            return {
                "overall_score": 0.0,
                "grade": "F",
                "components": {
                    "completeness": {"score": 0.0, "max": cls.WEIGHTS["completeness"], "deduction": cls.WEIGHTS["completeness"]},
                    "uniqueness": {"score": 0.0, "max": cls.WEIGHTS["uniqueness"], "deduction": cls.WEIGHTS["uniqueness"]},
                    "consistency": {"score": 0.0, "max": cls.WEIGHTS["consistency"], "deduction": cls.WEIGHTS["consistency"]},
                    "integrity": {"score": 0.0, "max": cls.WEIGHTS["integrity"], "deduction": cls.WEIGHTS["integrity"]},
                },
                "deductions": [{"category": "structure", "points": 100.0, "reason": "Dataset is empty."}],
                "top_impacts": ["Dataset has zero usable rows or columns."],
                "methodology": "The Data Quality Score evaluates Completeness (35%), Uniqueness (25%), Consistency (20%), and Structural Integrity (20%).",
            }

        deductions: List[Dict[str, Any]] = []

        # 1. Completeness Score (Max 35)
        missing_pct = profile_result.get("missing_pct", 0.0)
        # Non-linear penalty: mild for <5%, steep above 20%
        completeness_penalty = min(cls.WEIGHTS["completeness"], (missing_pct / 100.0) * cls.WEIGHTS["completeness"] * 1.5)
        completeness_score = max(0.0, round(cls.WEIGHTS["completeness"] - completeness_penalty, 2))
        if completeness_penalty > 0:
            deductions.append({
                "category": "completeness",
                "points": round(completeness_penalty, 2),
                "reason": f"{missing_pct}% of total dataset cells are missing or null."
            })

        # 2. Uniqueness Score (Max 25)
        dup_pct = profile_result.get("duplicate_pct", 0.0)
        uniqueness_penalty = min(cls.WEIGHTS["uniqueness"], (dup_pct / 100.0) * cls.WEIGHTS["uniqueness"] * 2.0)
        uniqueness_score = max(0.0, round(cls.WEIGHTS["uniqueness"] - uniqueness_penalty, 2))
        if uniqueness_penalty > 0:
            deductions.append({
                "category": "uniqueness",
                "points": round(uniqueness_penalty, 2),
                "reason": f"{dup_pct}% of dataset records are identical duplicate rows."
            })

        # 3. Consistency Score (Max 20)
        consistency_penalty = 0.0
        constant_cols = profile_result.get("constant_cols", [])
        if constant_cols and total_cols > 0:
            const_pen = (len(constant_cols) / total_cols) * 10.0
            consistency_penalty += const_pen
            deductions.append({
                "category": "consistency",
                "points": round(const_pen, 2),
                "reason": f"{len(constant_cols)} column(s) have constant, zero-variance values."
            })

        # Whitespace / data type inconsistencies
        issues = profile_result.get("issues", [])
        ws_issues = [i for i in issues if i.get("issue_type") == "Whitespace Inconsistency"]
        if ws_issues:
            consistency_penalty += min(5.0, len(ws_issues) * 1.5)
            deductions.append({
                "category": "consistency",
                "points": min(5.0, len(ws_issues) * 1.5),
                "reason": f"{len(ws_issues)} text column(s) contain un-trimmed leading/trailing spaces."
            })

        type_issues = [i for i in issues if "as Text" in i.get("issue_type", "")]
        if type_issues:
            consistency_penalty += min(5.0, len(type_issues) * 2.0)
            deductions.append({
                "category": "consistency",
                "points": min(5.0, len(type_issues) * 2.0),
                "reason": f"{len(type_issues)} column(s) appear to store numeric or date values as raw strings."
            })

        consistency_penalty = min(cls.WEIGHTS["consistency"], consistency_penalty)
        consistency_score = max(0.0, round(cls.WEIGHTS["consistency"] - consistency_penalty, 2))

        # 4. Integrity Score (Max 20)
        integrity_penalty = 0.0
        empty_row_issues = [i for i in issues if i.get("issue_type") == "Empty Rows"]
        if empty_row_issues:
            integrity_penalty += 8.0
            deductions.append({
                "category": "integrity",
                "points": 8.0,
                "reason": "Completely empty rows detected."
            })

        empty_col_issues = [i for i in issues if i.get("issue_type") == "Empty Column"]
        if empty_col_issues:
            integrity_penalty += min(10.0, len(empty_col_issues) * 5.0)
            deductions.append({
                "category": "integrity",
                "points": min(10.0, len(empty_col_issues) * 5.0),
                "reason": f"{len(empty_col_issues)} column(s) contain 100% missing values."
            })

        integrity_penalty = min(cls.WEIGHTS["integrity"], integrity_penalty)
        integrity_score = max(0.0, round(cls.WEIGHTS["integrity"] - integrity_penalty, 2))

        overall_score = round(completeness_score + uniqueness_score + consistency_score + integrity_score, 1)

        # Assign Grade
        if overall_score >= 90:
            grade = "A (Excellent)"
        elif overall_score >= 80:
            grade = "B (Good)"
        elif overall_score >= 70:
            grade = "C (Fair)"
        elif overall_score >= 50:
            grade = "D (Needs Attention)"
        else:
            grade = "F (Poor)"

        # Sort top impact issues
        sorted_deductions = sorted(deductions, key=lambda x: x["points"], reverse=True)
        top_impacts = [f"-{d['points']} pts: {d['reason']}" for d in sorted_deductions[:4]]

        return {
            "overall_score": overall_score,
            "grade": grade,
            "components": {
                "completeness": {
                    "score": completeness_score,
                    "max": cls.WEIGHTS["completeness"],
                    "deduction": round(completeness_penalty, 2),
                },
                "uniqueness": {
                    "score": uniqueness_score,
                    "max": cls.WEIGHTS["uniqueness"],
                    "deduction": round(uniqueness_penalty, 2),
                },
                "consistency": {
                    "score": consistency_score,
                    "max": cls.WEIGHTS["consistency"],
                    "deduction": round(consistency_penalty, 2),
                },
                "integrity": {
                    "score": integrity_score,
                    "max": cls.WEIGHTS["integrity"],
                    "deduction": round(integrity_penalty, 2),
                },
            },
            "deductions": deductions,
            "top_impacts": top_impacts,
            "methodology": "The Data Quality Score is a reproducible metric calculated from Completeness (35%), Uniqueness (25%), Consistency (20%), and Structural Integrity (20%). It serves as an explainable heuristic indicator of analysis-readiness.",
        }
