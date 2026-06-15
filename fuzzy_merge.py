import difflib
import math
import re
import pandas as pd
from typing import List, Optional, Tuple, Union


def _levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def _jaccard_similarity(s1: str, s2: str, ngram: int = 2) -> float:
    def get_ngrams(text: str, n: int) -> set:
        text = re.sub(r'\s+', '', text)
        if len(text) < n:
            return {text}
        return set(text[i:i + n] for i in range(len(text) - n + 1))

    set1 = get_ngrams(s1, ngram)
    set2 = get_ngrams(s2, ngram)
    if not set1 and not set2:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def _cosine_similarity(s1: str, s2: str, ngram: int = 2) -> float:
    def get_ngram_vector(text: str, n: int) -> dict:
        text = re.sub(r'\s+', '', text)
        vec = {}
        if len(text) < n:
            vec[text] = 1
            return vec
        for i in range(len(text) - n + 1):
            gram = text[i:i + n]
            vec[gram] = vec.get(gram, 0) + 1
        return vec

    vec1 = get_ngram_vector(s1, ngram)
    vec2 = get_ngram_vector(s2, ngram)

    all_grams = set(vec1.keys()) | set(vec2.keys())
    dot_product = sum(vec1.get(gram, 0) * vec2.get(gram, 0) for gram in all_grams)
    norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def _edit_distance_similarity(s1: str, s2: str) -> float:
    distance = _levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - (distance / max_len)


def calculate_similarity(
    str1: str,
    str2: str,
    ignore_case: bool = True,
    algorithm: str = "difflib",
) -> float:
    if not isinstance(str1, str) or not isinstance(str2, str):
        return 0.0
    if ignore_case:
        str1 = str1.lower()
        str2 = str2.lower()

    algorithm = algorithm.lower()
    if algorithm == "difflib":
        return difflib.SequenceMatcher(None, str1, str2).ratio()
    elif algorithm == "levenshtein" or algorithm == "edit_distance":
        return _edit_distance_similarity(str1, str2)
    elif algorithm == "jaccard":
        return _jaccard_similarity(str1, str2)
    elif algorithm == "cosine":
        return _cosine_similarity(str1, str2)
    else:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            f"Use 'difflib', 'levenshtein'/'edit_distance', 'jaccard', or 'cosine'."
        )


def find_best_match(
    target: str,
    candidates: List[str],
    threshold: float = 0.8,
    ignore_case: bool = True,
    top_n: int = 1,
    algorithm: str = "difflib",
) -> Union[Optional[Tuple[str, float]], List[Tuple[str, float]]]:
    if not isinstance(target, str) or not candidates:
        return [] if top_n > 1 else None

    matches = []
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        score = calculate_similarity(target, candidate, ignore_case, algorithm)
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
    threshold: float = 0.8,
    ignore_case: bool = True,
    how: str = "left",
    include_similarity: bool = True,
    suffixes: Tuple[str, str] = ("_left", "_right"),
    algorithm: str = "difflib",
) -> pd.DataFrame:
    if left_on not in left_df.columns:
        raise ValueError(f"Column '{left_on}' not found in left DataFrame")
    if right_on not in right_df.columns:
        raise ValueError(f"Column '{right_on}' not found in right DataFrame")

    if how not in ["left", "right", "inner", "outer"]:
        raise ValueError(f"Invalid merge type '{how}'. Use 'left', 'right', 'inner', or 'outer'")

    valid_algorithms = ["difflib", "levenshtein", "edit_distance", "jaccard", "cosine"]
    if algorithm.lower() not in valid_algorithms:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            f"Use one of: {', '.join(valid_algorithms)}"
        )

    right_candidates = right_df[right_on].dropna().astype(str).tolist()

    match_results = []

    for idx, left_val in left_df[left_on].items():
        best_match = find_best_match(
            str(left_val) if pd.notna(left_val) else "",
            right_candidates,
            threshold=threshold,
            ignore_case=ignore_case,
            top_n=1,
            algorithm=algorithm,
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
                    result_row["algorithm"] = algorithm
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
                    result_row["algorithm"] = algorithm
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
                    result_row["algorithm"] = algorithm
                match_results.append(result_row)

    result_df = pd.DataFrame(match_results)
    return result_df


def fuzzy_match_report(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_on: str,
    right_on: str,
    threshold: float = 0.8,
    ignore_case: bool = True,
    algorithm: str = "difflib",
) -> pd.DataFrame:
    if left_on not in left_df.columns:
        raise ValueError(f"Column '{left_on}' not found in left DataFrame")
    if right_on not in right_df.columns:
        raise ValueError(f"Column '{right_on}' not found in right DataFrame")

    valid_algorithms = ["difflib", "levenshtein", "edit_distance", "jaccard", "cosine"]
    if algorithm.lower() not in valid_algorithms:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            f"Use one of: {', '.join(valid_algorithms)}"
        )

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
            algorithm=algorithm,
        )
        row = {
            left_on: left_val,
            "algorithm": algorithm,
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
