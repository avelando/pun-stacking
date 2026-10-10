from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from pun_stacking.graphs.builders import normalize_tokens


@dataclass
class GraphFeatureEncoder:
    node_index: dict[str, int]
    node_vectors: np.ndarray

    @classmethod
    def fit(
        cls,
        graph: nx.Graph,
        dimensions: int = 4,
        seed: int = 42,
    ) -> "GraphFeatureEncoder":
        nodes = sorted(graph.nodes())

        if len(nodes) <= dimensions:
            raise ValueError("Graph must contain more nodes than SVD dimensions")

        adjacency = csr_matrix(
            nx.to_scipy_sparse_array(
                graph,
                nodelist=nodes,
                weight="weight",
                dtype=np.float64,
                format="csr",
            )
        )

        svd = TruncatedSVD(
            n_components=dimensions,
            random_state=seed,
        )

        vectors = svd.fit_transform(adjacency).astype(np.float32)

        return cls(
            node_index={node: index for index, node in enumerate(nodes)},
            node_vectors=vectors,
        )

    def transform(self, token_sequences: list[list[str]]) -> np.ndarray:
        dimensions = self.node_vectors.shape[1]

        features = np.zeros(
            (len(token_sequences), dimensions + 2),
            dtype=np.float32,
        )

        for row_index, raw_tokens in enumerate(token_sequences):
            tokens = normalize_tokens(raw_tokens)

            if not tokens:
                continue

            indices = [self.node_index[token] for token in tokens if token in self.node_index]

            hits = len(indices)

            if hits:
                features[row_index, :dimensions] = self.node_vectors[indices].mean(axis=0)

            features[row_index, dimensions] = hits / len(tokens)
            features[row_index, dimensions + 1] = np.log1p(hits)

        return features
