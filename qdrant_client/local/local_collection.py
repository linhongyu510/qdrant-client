import json
import math
import uuid
from collections import OrderedDict, defaultdict
from typing import (
    Any,
    Callable,
    Mapping,
    Sequence,
    get_args,
)
from copy import deepcopy
import warnings

import numpy as np

from qdrant_client import grpc as grpc
from qdrant_client.common.client_warnings import show_warning_once
from qdrant_client._pydantic_compat import (
    construct,
    to_dict,
    to_jsonable_python as _to_jsonable_python,
)
from qdrant_client.conversions import common_types as types
from qdrant_client.conversions.common_types import get_args_subscribed
from qdrant_client.conversions.conversion import GrpcToRest
from qdrant_client.http import models
from qdrant_client.http.models import ScoredPoint
from qdrant_client.http.models.models import Distance, ExtendedPointId, SparseVector, OrderValue
from qdrant_client.hybrid.formula import evaluate_expression, raise_non_finite_error
from qdrant_client.hybrid.fusion import reciprocal_rank_fusion, distribution_based_score_fusion
from qdrant_client.local.distances import (
    ContextPair,
    ContextQuery,
    DenseQueryVector,
    DiscoveryQuery,
    DistanceOrder,
    RecoQuery,
    calculate_context_scores,
    calculate_discovery_scores,
    calculate_distance,
    calculate_recommend_best_scores,
    distance_to_order,
    calculate_recommend_sum_scores,
    calculate_distance_core,
    calculate_naive_feedback_query,
    NaiveFeedbackQuery,
    FeedbackItem as DistFeedbackItem,
    NaiveFeedbackCoefficients,
)
from qdrant_client.local.multi_distances import (
    MultiQueryVector,
    MultiRecoQuery,
    MultiDiscoveryQuery,
    MultiContextQuery,
    MultiContextPair,
    calculate_multi_distance,
    calculate_multi_recommend_best_scores,
    calculate_multi_discovery_scores,
    calculate_multi_context_scores,
    calculate_multi_recommend_sum_scores,
    calculate_multi_distance_core,
)
from qdrant_client.local.json_path_parser import JsonPathItem, parse_json_path
from qdrant_client.local.order_by import to_order_value
from qdrant_client.local.payload_filters import (
    calculate_payload_mask,
    check_filter,
    validate_filter,
)
from qdrant_client.local.payload_value_extractor import value_by_key, parse_uuid
from qdrant_client.local.payload_value_setter import delete_value_by_key, set_value_by_key
from qdrant_client.local.persistence import CollectionPersistence
from qdrant_client.local.utils import last_argmax, swap_remove
from qdrant_client.local.sparse import (
    copy_sparse_vector,
    empty_sparse_vector,
    sort_sparse_vector,
    validate_sparse_vector,
)
from qdrant_client.local.sparse_distances import (
    SparseContextPair,
    SparseContextQuery,
    SparseDiscoveryQuery,
    SparseQueryVector,
    SparseRecoQuery,
    calculate_distance_sparse,
    calculate_sparse_context_scores,
    calculate_sparse_discovery_scores,
    calculate_sparse_recommend_best_scores,
    merge_positive_and_negative_avg,
    sparse_avg,
    calculate_sparse_recommend_sum_scores,
)

DEFAULT_VECTOR_NAME = ""
EPSILON = 1.1920929e-7  # https://doc.rust-lang.org/std/f32/constant.EPSILON.html
# https://github.com/qdrant/qdrant/blob/7164ac4a5987d28f1c93f5712aef8e09e7d93555/lib/segment/src/spaces/simple_avx.rs#L99C10-L99C10


def to_jsonable_python(x: Any) -> Any:
    try:
        return json.loads(json.dumps(x, allow_nan=True))
    except Exception:
        return json.loads(json.dumps(x, allow_nan=True, default=_to_jsonable_python))


def validate_dense_vector(vector: Any, vector_name: str) -> None:
    """Reject empty dense vectors and NaN values, as the server does at write time."""
    if len(vector) == 0:
        raise ValueError(f"Wrong input: Dense vector must not be empty for vector '{vector_name}'")

    if np.isnan(np.asarray(vector, dtype=np.float32)).any():
        raise ValueError("Vector contains NaN values")


def validate_vector_dimension(got: int, expected: int, vector_name: str) -> None:
    """Reject a vector whose size does not match the collection's, as the server does.

    Without this a wrong-size dense vector reached numpy and died with a raw broadcast
    error part-way through the write, and a wrong-size multivector was stored silently.
    """
    if got != expected:
        raise ValueError(
            f"Wrong input: Vector dimension error: expected dim: {expected}, "
            f"got {got} for vector '{vector_name}'"
        )


def validate_multivector(vector: Any, vector_name: str) -> None:
    """Reject empty multivectors, empty sub-vectors and NaN values, as the server does."""
    if len(vector) == 0:
        raise ValueError(f"Wrong input: Multivector must not be empty for vector '{vector_name}'")

    for sub_vector in vector:
        if hasattr(sub_vector, "__len__") and len(sub_vector) == 0:
            raise ValueError(
                "Wrong input: All vectors of a multivector must be non-empty "
                f"for vector '{vector_name}'"
            )

    if np.isnan(np.asarray(vector, dtype=np.float32)).any():
        raise ValueError("Vector contains NaN values")


class LocalCollection:
    """
    LocalCollection is a class that represents a collection of vectors in the local storage.
    """

    LARGE_DATA_THRESHOLD = 20_000

    def __init__(
        self,
        config: models.CreateCollection,
        location: str | None = None,
        force_disable_check_same_thread: bool = False,
    ) -> None:
        """
        Create or load a collection from the local storage.
        Args:
            location: path to the collection directory. If None, the collection will be created in memory.
            force_disable_check_same_thread: force disable check_same_thread for sqlite3 connection. default: False
        """
        # copy before deriving `vectors_config`, so the collection shares nothing with the caller
        config = deepcopy(config)
        self.vectors_config, self.multivectors_config = self._resolve_vectors_config(
            config.vectors
        )
        sparse_vectors_config = config.sparse_vectors
        self.vectors: dict[str, types.NumpyArray] = {
            name: np.zeros((0, params.size), dtype=np.float32)
            for name, params in self.vectors_config.items()
        }
        self.sparse_vectors: dict[str, list[SparseVector]] = (
            {name: [] for name, params in sparse_vectors_config.items()}
            if sparse_vectors_config is not None
            else {}
        )
        self.sparse_vectors_idf: dict[
            str, dict[int, int]
        ] = {}  # vector_name: {idx_in_vocab: doc frequency}
        self.multivectors: dict[str, list[types.NumpyArray]] = {
            name: [] for name in self.multivectors_config
        }
        self.payload: list[models.Payload] = []
        self.deleted: types.NumpyArray = np.zeros(0, dtype=bool)
        self._all_vectors_keys = (
            list(self.vectors.keys())
            + list(self.sparse_vectors.keys())
            + list(self.multivectors.keys())
        )
        self.deleted_per_vector: dict[str, types.NumpyArray] = {
            name: np.zeros(0, dtype=bool) for name in self._all_vectors_keys
        }
        self.ids: dict[models.ExtendedPointId, int] = {}  # Mapping from external id to internal id
        self.ids_inv: list[models.ExtendedPointId] = []  # Mapping from internal id to external id
        self.persistent = location is not None
        self.storage = None
        self.config = config
        if location is not None:
            self.storage = CollectionPersistence(location, force_disable_check_same_thread)
        self.load_vectors()

    @staticmethod
    def _resolve_vectors_config(
        vectors: dict[str, models.VectorParams],
    ) -> tuple[dict[str, models.VectorParams], dict[str, models.VectorParams]]:
        vectors_config = {}
        multivectors_config = {}
        if isinstance(vectors, models.VectorParams):
            if vectors.multivector_config is not None:
                multivectors_config = {DEFAULT_VECTOR_NAME: vectors}
            else:
                vectors_config = {DEFAULT_VECTOR_NAME: vectors}
            return vectors_config, multivectors_config

        for name, params in vectors.items():
            if params.multivector_config is not None:
                multivectors_config[name] = params
            else:
                vectors_config[name] = params

        return vectors_config, multivectors_config

    def close(self) -> None:
        if self.storage is not None:
            self.storage.close()

    def _update_idf_append(self, vector: SparseVector, vector_name: str) -> None:
        if vector_name not in self.sparse_vectors_idf:
            self.sparse_vectors_idf[vector_name] = defaultdict(int)
        for idx in vector.indices:
            self.sparse_vectors_idf[vector_name][idx] += 1

    def _update_idf_remove(self, vector: SparseVector, vector_name: str) -> None:
        for idx in vector.indices:
            self.sparse_vectors_idf[vector_name][idx] -= 1

    def _drop_idf_contribution(self, idx: int, vector_name: str) -> None:
        """Take point `idx` out of the IDF counters of `vector_name`.

        `sparse_vectors_idf` must stay in sync with the corpus `_rescore_idf` measures, which
        is every point that is alive and actually has the vector. A point that is already
        deleted, point-wise or vector-wise, never counted, so dropping it again is a no-op.
        """
        if self.deleted[idx] or self.deleted_per_vector[vector_name][idx]:
            return
        self._update_idf_remove(self.sparse_vectors[vector_name][idx], vector_name)

    def _existing_idx(self, point_id: types.PointId) -> int | None:
        """Internal id of a point the collection still holds, `None` if it holds none.

        Deleted points keep their slot so that internal ids stay stable, but the server
        answers 404 for them exactly as it does for ids it has never seen.
        """
        idx = self.ids.get(point_id)
        if idx is None or self.deleted[idx]:
            return None
        return idx

    @staticmethod
    def _idf_corpus_of(search_params: types.SearchParams | None) -> types.Filter | None:
        """Corpus filter scoping IDF statistics, if `search_params` asks for a narrowed scope.

        `IdfScope.GLOBAL` (and no `idf` at all) means collection-wide statistics, which is
        represented by `None`.
        """
        if search_params is None:
            return None
        if isinstance(search_params.idf, models.IdfCorpusParams):
            return search_params.idf.corpus
        return None

    @classmethod
    def _compute_idf(cls, df: int, n: int) -> float:
        # ((n - df + 0.5) / (df + 0.5) + 1.).ln()
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def _corpus_document_frequencies(
        self, vector_name: str, indices: Sequence[int], mask: np.ndarray
    ) -> dict[int, int]:
        """Document frequency of each of `indices` among the points selected by `mask`."""
        wanted = {int(idx) for idx in indices}
        frequencies = dict.fromkeys(wanted, 0)

        vectors = self.sparse_vectors[vector_name]
        for internal_id in np.nonzero(mask)[0]:
            if internal_id >= len(vectors):
                continue
            for idx in vectors[internal_id].indices:
                if idx in wanted:
                    frequencies[idx] += 1

        return frequencies

    def _rescore_idf(
        self,
        vector: SparseVector,
        vector_name: str,
        idf_corpus: types.Filter | None = None,
    ) -> SparseVector:
        idf_store = self.sparse_vectors_idf.get(vector_name)
        if idf_store is None:
            return vector

        # IDF statistics only take into account points which actually have this sparse vector,
        # points missing it are not part of the corpus. `idf_corpus` narrows it down further.
        #
        # Deleted points leave the corpus right away. `IdfScope.GLOBAL` on the server reads
        # the sparse index rather than the live points, so it goes on counting deleted ones
        # for as long as they sit in that index; local mode has no index to go stale and
        # answers from live statistics immediately. The two agree on `IdfCorpusParams`, which
        # is measured over live points either way - that is what the congruence tests pin.
        mask = self._payload_and_non_deleted_mask(idf_corpus, vector_name=vector_name)
        num_docs = int(np.count_nonzero(mask))

        document_frequencies: Mapping[int, int] = (
            idf_store
            if idf_corpus is None
            else self._corpus_document_frequencies(vector_name, vector.indices, mask)
        )

        new_values = []
        for idx, value in zip(vector.indices, vector.values):
            document_frequency = document_frequencies.get(idx, 0)
            idf = self._compute_idf(document_frequency, num_docs)
            new_values.append(value * idf)

        return SparseVector(indices=vector.indices, values=new_values)

    def load_vectors(self) -> None:
        if self.storage is not None:
            vectors = defaultdict(list)
            sparse_vectors = defaultdict(list)
            multivectors = defaultdict(list)
            deleted_ids = []

            for idx, point in enumerate(self.storage.load()):
                # id tracker
                self.ids[point.id] = idx
                # no gaps in idx
                self.ids_inv.append(point.id)

                # payload tracker
                self.payload.append(to_jsonable_python(point.payload) or {})

                # persisted named vectors
                loaded_vector = point.vector

                # add default name to anonymous dense or multivector
                if isinstance(point.vector, list):
                    loaded_vector = {DEFAULT_VECTOR_NAME: point.vector}

                # handle dense vectors
                all_dense_vector_names = list(self.vectors.keys())
                for name in all_dense_vector_names:
                    v = loaded_vector.get(name)
                    if v is not None:
                        vectors[name].append(v)
                    else:
                        vectors[name].append(
                            np.ones(self.vectors_config[name].size, dtype=np.float32)
                        )
                        deleted_ids.append((idx, name))

                # handle sparse vectors
                all_sparse_vector_names = list(self.sparse_vectors.keys())
                for name in all_sparse_vector_names:
                    v = loaded_vector.get(name)
                    if v is not None:
                        sparse_vectors[name].append(v)
                    else:
                        sparse_vectors[name].append(empty_sparse_vector())
                        deleted_ids.append((idx, name))

                # handle multivectors
                all_multivector_names = list(self.multivectors.keys())
                for name in all_multivector_names:
                    v = loaded_vector.get(name)
                    if v is not None:
                        multivectors[name].append(v)
                    else:
                        multivectors[name].append(
                            np.ones((1, self.multivectors_config[name].size), dtype=np.float32)
                        )
                        deleted_ids.append((idx, name))

            # setup dense vectors by name
            for name, named_vectors in vectors.items():
                self.vectors[name] = np.array(named_vectors)
                self.deleted_per_vector[name] = np.zeros(len(self.payload), dtype=bool)

            # setup sparse vectors by name
            for name, named_vectors in sparse_vectors.items():
                self.sparse_vectors[name] = named_vectors
                self.deleted_per_vector[name] = np.zeros(len(self.payload), dtype=bool)
                for vector in named_vectors:
                    self._update_idf_append(vector, name)

            # setup multivectors by name
            for name, named_vectors in multivectors.items():
                self.multivectors[name] = [np.array(vector) for vector in named_vectors]
                self.deleted_per_vector[name] = np.zeros(len(self.payload), dtype=bool)

            # track deleted points by named vector
            for idx, name in deleted_ids:
                self.deleted_per_vector[name][idx] = 1

            self.deleted = np.zeros(len(self.payload), dtype=bool)