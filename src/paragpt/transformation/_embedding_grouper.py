import math
import math
import pathlib as pt
from collections import deque
from os.path import exists
from typing import List

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from sklearn.metrics.pairwise import cosine_similarity

from summargpt.summarize import read_from_file, save_to_file


def _rev_sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(0.5 * x))


def _activate_similarities(similarities: np.array, n_ticks: int = 9) -> np.array:
    n_ticks = min(n_ticks, similarities.shape[0])
    x = np.linspace(-10, 10, n_ticks)
    y = np.vectorize(_rev_sigmoid)
    activation_weights = np.pad(y(x), (0, similarities.shape[0] - n_ticks))
    diagonals = [
        similarities.diagonal(each) for each in range(0, similarities.shape[0])
    ]
    diagonals = [
        np.pad(each, (0, similarities.shape[0] - len(each))) for each in diagonals
    ]
    diagonals = np.stack(diagonals)
    diagonals = diagonals * activation_weights.reshape(-1, 1)
    activated_similarities = np.sum(diagonals, axis=0)
    return activated_similarities


def _grouper(
    df: pd.DataFrame,
    char: str,
    n_ticks: int = 100,
    embedding_name="embedding",
    snippet_name="snippet",
    n_token_name="n_tokens",
    group_weight: int = 1500,
    order=2
) -> pd.DataFrame:
    embeddings: pd.Series
    snippets: pd.Series

    embeddings = df[embedding_name]
    snippets = df[snippet_name]

    similarities = cosine_similarity(np.array(embeddings.to_list()))
    activated_similarities = _activate_similarities(similarities, n_ticks=n_ticks)
    minmimas = argrelextrema(activated_similarities, np.less, order=order)[0].tolist()
    minmimas = deque(minmimas)

    if not (len(df) in minmimas):
        minmimas.append(len(snippets))  # A group to collect all remaining snippets

    group_assignment = []

    current_minima = minmimas.popleft()
    token_sum = 0
    rows_to_replace = {}
    for i in df.index:
        while i == current_minima:
            # due to snippets sometimes needing to be split up
            # the minima might need to increase more than once
            current_minima = minmimas.popleft()
            token_sum = 0
        tokens_for_this_snippet = df.at[i, n_token_name]
        # Some snippets (one uninterrupted speaker) are too long in themselves already.
        # Split those snippets into approximate halfs (by sentence).
        # Not an accurate science, but beats failing.
        if tokens_for_this_snippet > group_weight:
            snippet = df.at[i, snippet_name]
            sentences = snippet.split(".")
            divider = math.ceil(tokens_for_this_snippet / group_weight)
            tokens_for_this_snippet = tokens_for_this_snippet/divider
            for j in range(divider):
                if j+1 == divider:
                    # take all sentences that are left, necessary in case the division isn't equal
                    chunk = sentences
                else:
                    sentences_to_take = int(len(sentences) / divider)
                    chunk = sentences[:sentences_to_take]
                    sentences = sentences[sentences_to_take:]
                entry = df.loc[i]
                entry[snippet_name] = ".".join(chunk)
                entry[n_token_name] = tokens_for_this_snippet
                if i in rows_to_replace:
                    rows_to_replace[i].append(entry)
                else:
                    rows_to_replace[i] = [entry]

                # Put each of them into their own group
                current_minima += 1
                group_assignment.append(current_minima)
            token_sum = tokens_for_this_snippet

        else :
            token_sum += tokens_for_this_snippet
            # It might be that the sum of tokens for the grouped snippets is too big,
            # so prevent them from being grouped together.
            if token_sum > group_weight:
                current_minima += 1
                token_sum = df.at[i, n_token_name]
            group_assignment.append(current_minima)

    num_new_rows = 0
    for r in rows_to_replace:
        replacements = rows_to_replace[r]
        index_in_df = r + num_new_rows
        df = df.drop(index_in_df)
        new_rows = pd.DataFrame(replacements)
        df = pd.concat([df.iloc[:index_in_df], new_rows, df.iloc[index_in_df:]]).reset_index(drop=True)
        num_new_rows += (len(replacements) - 1)
    df["group"] = group_assignment

    group_by = df.groupby("group")

    grouped_snippets = group_by[snippet_name].apply(lambda x: "\n".join(x)).reset_index(drop=True)
    grouped_tokens = group_by[n_token_name].sum().reset_index(drop=True)

    snippet_n_tokens = pd.DataFrame({
        snippet_name: grouped_snippets,
        n_token_name: grouped_tokens
    })

    return snippet_n_tokens


def group_by_embedding(df: pd.DataFrame, order=2, group_weight=1200) -> List[str]:
    snippet_n_token_file = "snippet_n_token_embeddings.txt"
    if exists(snippet_n_token_file):
        snippet_n_tokens = read_from_file(snippet_n_token_file)
    else:
        snippet_n_tokens = _grouper(df.copy(), "\n", group_weight=group_weight, order=order)
        save_to_file(snippet_n_tokens, snippet_n_token_file)

    new_groupings = []
    current_group = 0
    current_weight = 0
    for weight in snippet_n_tokens["n_tokens"]:
        current_weight+=weight
        if current_weight > group_weight:
            current_weight = 0
            new_groupings.append(current_group)
            current_group += 1
        else:
            new_groupings.append(current_group)
    snippet_n_tokens["group"] = new_groupings
    output = snippet_n_tokens.groupby("group")["snippet"].apply(lambda x : "\n".join(x)).to_list()

    return output


if __name__ == "__main__":
    file_path = pt.Path("../../../df_out.csv")
    df = pd.read_csv(file_path, index_col=0, dtype={"embedding": object})
    df["embedding"] = df["embedding"].apply(eval)

    group_by_embedding(df, group_weight=1500)
