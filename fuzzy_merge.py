import difflib
import pandas as pd
from typing import List, Optional, Tuple, Union


def calculate_similarity(str1: str, str2: str, ignore_case: bool = True) -> float:
    if not isinstance(str1, str) or not isinstance(str2, str):
        return 0.0
    if ignore_case:
        str1 = str1.lower()
        str2 = str2.lower()
    return difflib.SequenceMatcher(None, str1, str2).ratio()


def find_best_match(
    target: str,
    candidates: List[str],
    threshold: float = 0.6,
    ignore_case: bool = True,
    top_n: int = 1,
) -> Union[Optional[Tuple[str, float]], List[Tuple[str, float]]]:
    if not isinstance(target, str) or not candidates:
        return [] if top_n > 1 else None

    matches = []
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        score = calculate_similarity(target, candidate, ignore_case)
        if score >= threshold:
            matches.append((candidate, score))

    matches.sort(key=lambda x: x[1], reverse=True)

    if top_n == 1:
        return matches[0] if matches else None
    return matches[:top_n]


def fuzzy_merge(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_on: str,
    right_on: str,
    threshold: float = 0.6,
    ignore_case: bool = True,
    how: str = "left",
    include_similarity: bool = True,
    suffixes: Tuple[str, str] = ("_left", "_right"),
) -> pd.DataFrame:
    if left_on not in left_df.columns:
        raise ValueError(f"Column '{left_on}' not found in left DataFrame")
    if right_on not in right_df.columns:
        raise ValueError(f"Column '{right_on}' not found in right DataFrame")

    if how not in ["left", "right", "inner", "outer"]:
        raise ValueError(f"Invalid merge type '{how}'. Use 'left', 'right', 'inner', or 'outer'")

    right_candidates = right_df[right_on].dropna().astype(str).tolist()

    match_results = []

    for idx, left_val in left_df[left_on].items():
        best_match = find_best_match(
            str(left_val) if pd.notna(left_val) else "",
            right_candidates,
            threshold=threshold,
            ignore_case=ignore_case,
            top_n=1,
        )
        if best_match:
            matched_str, score = best_match
            matched_rows = right_df[right_df[right_on].astype(str) == matched_str]
            for _, right_row in matched_rows.iterrows():
                result_row = {}
                for col in left_df.columns:
                    result_row[col] = left_df.loc[idx, col]
                for col in right_df.columns:
                    col_name = col if col not in left_df.columns else col + suffixes[1]
                    result_row[col_name] = right_row[col]
                if include_similarity:
                    result_row["similarity"] = round(score, 4)
                match_results.append(result_row)
        else:
            if how in ["left", "outer"]:
                result_row = {}
                for col in left_df.columns:
                    result_row[col] = left_df.loc[idx, col]
                for col in right_df.columns:
                    col_name = col if col not in left_df.columns else col + suffixes[1]
                    result_row[col_name] = pd.NA
                if include_similarity:
                    result_row["similarity"] = pd.NA
                match_results.append(result_row)

    if how in ["right", "outer"]:
        matched_right_values = set()
        for row in match_results:
            if right_on in row and pd.notna(row[right_on]):
                matched_right_values.add(str(row[right_on]))

        for idx, right_val in right_df[right_on].items():
            right_val_str = str(right_val) if pd.notna(right_val) else ""
            if right_val_str not in matched_right_values:
                result_row = {}
                for col in left_df.columns:
                    col_name = col if col not in right_df.columns else col + suffixes[0]
                    result_row[col_name] = pd.NA
                for col in right_df.columns:
                    result_row[col] = right_df.loc[idx, col]
                if include_similarity:
                    result_row["similarity"] = pd.NA
                match_results.append(result_row)

    result_df = pd.DataFrame(match_results)
    return result_df


def fuzzy_match_report(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_on: str,
    right_on: str,
    threshold: float = 0.6,
    ignore_case: bool = True,
) -> pd.DataFrame:
    if left_on not in left_df.columns:
        raise ValueError(f"Column '{left_on}' not found in left DataFrame")
    if right_on not in right_df.columns:
        raise ValueError(f"Column '{right_on}' not found in right DataFrame")

    right_candidates = right_df[right_on].dropna().astype(str).tolist()

    report_rows = []
    for idx, left_val in left_df[left_on].items():
        left_str = str(left_val) if pd.notna(left_val) else ""
        matches = find_best_match(
            left_str,
            right_candidates,
            threshold=threshold,
            ignore_case=ignore_case,
            top_n=5,
        )
        row = {
            left_on: left_val,
            "match_status": "matched" if matches else "no_match",
            "top_match": matches[0][0] if matches else None,
            "top_similarity": round(matches[0][1], 4) if matches else None,
        }
        for i in range(1, 6):
            if i <= len(matches):
                row[f"match_{i}"] = matches[i - 1][0]
                row[f"score_{i}"] = round(matches[i - 1][1], 4)
            else:
                row[f"match_{i}"] = None
                row[f"score_{i}"] = None
        report_rows.append(row)

    return pd.DataFrame(report_rows)
