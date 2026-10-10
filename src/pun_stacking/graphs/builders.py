import math
import re
from collections import Counter, defaultdict

import networkx as nx

TOKEN_PATTERN = re.compile(
    r"[^a-záàâãéèêíïóôõöúçñ]+",
    flags=re.IGNORECASE,
)

GRAPH_NAMES = ("cooccurrence", "ppmi", "pun_context")


def normalize_token(token: str) -> str:
    return TOKEN_PATTERN.sub("", token.lower().strip())


def normalize_tokens(tokens: list[str]) -> list[str]:
    normalized = [normalize_token(token) for token in tokens]
    return [token for token in normalized if len(token) > 1]


def count_cooccurrences(
    rows: list[dict],
    window: int,
    min_freq: int,
) -> dict[tuple[str, str], int]:
    if window < 1 or min_freq < 1:
        raise ValueError("Window and minimum frequency must be positive")

    documents = [normalize_tokens(row["tokens"]) for row in rows]

    frequencies = Counter(token for document in documents for token in document)

    vocabulary = {token for token, frequency in frequencies.items() if frequency >= min_freq}

    counts = Counter()

    for document in documents:
        filtered = [token for token in document if token in vocabulary]

        for index, word in enumerate(filtered):
            start = max(0, index - window)
            stop = min(len(filtered), index + window + 1)

            for position in range(start, stop):
                if position == index:
                    continue

                neighbor = filtered[position]

                if neighbor == word:
                    continue

                edge = tuple(sorted((word, neighbor)))
                counts[edge] += 1

    return dict(counts)


def select_top_neighbors(
    edges: dict[tuple[str, str], float],
    top_k: int | None,
) -> dict[tuple[str, str], float]:
    if top_k is None:
        return dict(sorted(edges.items()))

    if top_k < 1:
        raise ValueError("top_k must be positive")

    neighbors = defaultdict(list)

    for (source, target), weight in edges.items():
        neighbors[source].append((target, weight))
        neighbors[target].append((source, weight))

    selected = set()

    for node in sorted(neighbors):
        ranked = sorted(
            neighbors[node],
            key=lambda item: (-item[1], item[0]),
        )

        for neighbor, _ in ranked[:top_k]:
            selected.add(tuple(sorted((node, neighbor))))

    return {edge: edges[edge] for edge in sorted(selected)}


def calculate_ppmi(
    counts: dict[tuple[str, str], int],
) -> dict[tuple[str, str], float]:
    total = sum(counts.values())

    if total == 0:
        raise ValueError("No co-occurrence events available for PPMI")

    endpoints = Counter()

    for (source, target), count in counts.items():
        endpoints[source] += count
        endpoints[target] += count

    edges = {}

    for (source, target), count in counts.items():
        numerator = 2.0 * total * count
        denominator = endpoints[source] * endpoints[target]

        score = math.log(numerator / denominator)

        if score > 0:
            edges[(source, target)] = score

    return edges


def count_pun_context(
    rows: list[dict],
    min_freq: int,
) -> dict[tuple[str, str], int]:
    if min_freq < 1:
        raise ValueError("Minimum frequency must be positive")

    positive_rows = [row for row in rows if row["label"] == 1]

    if not positive_rows:
        raise ValueError("Pun-Context requires positive training examples")

    documents = [normalize_tokens(row["tokens"]) for row in positive_rows]

    frequencies = Counter(token for document in documents for token in document)

    vocabulary = {token for token, frequency in frequencies.items() if frequency >= min_freq}

    counts = Counter()

    for row, document in zip(positive_rows, documents, strict=True):
        tokens = row["tokens"]
        labels = row["labels"]

        if len(tokens) != len(labels):
            raise ValueError(f"Invalid annotations for {row['id']}")

        pun_tokens = [
            normalize_token(token)
            for token, label in zip(tokens, labels, strict=True)
            if label == 1
        ]

        pun_tokens = [token for token in pun_tokens if len(token) > 1 and token in vocabulary]

        context = [token for token in document if token in vocabulary]

        for pun_token in pun_tokens:
            for context_token in context:
                if pun_token == context_token:
                    continue

                edge = tuple(sorted((pun_token, context_token)))
                counts[edge] += 1

    return dict(counts)


def edges_to_graph(
    edges: dict[tuple[str, str], float],
) -> nx.Graph:
    graph = nx.Graph()

    for (source, target), weight in sorted(edges.items()):
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("Graph edge weights must be finite and positive")

        graph.add_edge(source, target, weight=float(weight))

    return graph


def build_graph_set(
    train_rows: list[dict],
    config: dict,
) -> dict[str, nx.Graph]:
    if not train_rows:
        raise ValueError("Graph construction requires training examples")

    expected_names = set(GRAPH_NAMES)

    if set(config["names"]) != expected_names:
        raise ValueError("Unexpected graph configuration")

    cooccurrence_config = config["cooccurrence"]
    ppmi_config = config["ppmi"]
    pun_context_config = config["pun_context"]

    cooccurrence_counts = count_cooccurrences(
        train_rows,
        window=cooccurrence_config["window"],
        min_freq=cooccurrence_config["min_freq"],
    )

    ppmi_counts = count_cooccurrences(
        train_rows,
        window=ppmi_config["window"],
        min_freq=ppmi_config["min_freq"],
    )

    pun_context_counts = count_pun_context(
        train_rows,
        min_freq=pun_context_config["min_freq"],
    )

    edge_sets = {
        "cooccurrence": select_top_neighbors(
            cooccurrence_counts,
            cooccurrence_config["top_k"],
        ),
        "ppmi": select_top_neighbors(
            calculate_ppmi(ppmi_counts),
            ppmi_config["top_k"],
        ),
        "pun_context": select_top_neighbors(
            pun_context_counts,
            pun_context_config["top_k"],
        ),
    }

    graphs = {name: edges_to_graph(edges) for name, edges in edge_sets.items()}

    for name, graph in graphs.items():
        if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
            raise ValueError(f"Empty graph: {name}")

    return graphs
