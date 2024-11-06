import numpy
import itertools

from scipy.sparse import lil_matrix


def get_matrix(num_users: int, num_items: int) -> lil_matrix:
    numpy.random.seed(42)
    return lil_matrix((num_users, num_items))


def get_random(num_users: int, num_items: int) -> lil_matrix:
    return lil_matrix(numpy.random.randint(low=1, high=6, size=(num_users, num_items)))


def get_common_indices(matrix: lil_matrix) -> numpy.ndarray:
    return [
        numpy.intersect1d(ar1=matrix[u, :].nonzero()[1], ar2=matrix[v, :].nonzero()[1])
        for (u, v) in itertools.combinations(range(matrix.shape[0]), 2)
    ]


def get_similarity_user(matrix: lil_matrix, u: int) -> numpy.ndarray:
    if u > matrix.shape[0]:
        raise ValueError(
            f"user {u} is out of bounds for matrix of shape {matrix.shape}"
        )

    u_indices = matrix[u, :].nonzero()[1]
    v_indices = [matrix[user, :].nonzero()[1] for user in range(matrix.shape[0])]

    common_indices = [
        numpy.intersect1d(ar1=v_indices, ar2=u_indices) for v_indices in v_indices
    ]

    u_ratings = [matrix[u, x].toarray() for x in common_indices]
    v_ratings = [
        matrix[y, x].toarray() for y, x in zip(range(matrix.shape[0]), common_indices)
    ]

    u_mean = [numpy.mean(x, axis=1) for x in u_ratings]
    v_mean = [numpy.mean(x, axis=1) for x in v_ratings]

    var_u = [numpy.sum((y - x) ** 2) for y, x in zip(u_ratings, u_mean)]
    var_v = [numpy.sum((y - x) ** 2) for y, x in zip(v_ratings, v_mean)]

    covariance = [
        numpy.sum((y - x) * (z - w))
        for y, x, z, w in zip(u_ratings, u_mean, v_ratings, v_mean)
    ]
    similarity = [c / numpy.sqrt(a * b) for c, a, b in zip(covariance, var_u, var_v)]

    return numpy.nan_to_num(similarity)


def get_neighbors_user(
    matrix: lil_matrix, u: int, k: int
) -> tuple[numpy.ndarray, numpy.ndarray]:
    similarities = get_similarity_user(matrix=matrix, u=u)
    neighbors = numpy.argsort(similarities)[-k:]
    return neighbors, similarities[neighbors]


def get_mean_ratings_user(matrix: lil_matrix, u: list[int]) -> numpy.ndarray:
    return numpy.mean(matrix[u, :].toarray(), axis=1)


def get_predictions_user(matrix: lil_matrix, u: int, k: int) -> numpy.ndarray:
    neighbors, similarities = get_neighbors_user(matrix=matrix, u=u, k=k)
    mean_ratings = get_mean_ratings_user(matrix=matrix, u=neighbors)

    scores = numpy.array(matrix[u, :].toarray()[0, :], dtype=float)
    for i in range(matrix.shape[1]):
        if scores[i] != 0:
            continue

        neighbors_rated_i = [
            (n, similarity, mean_rating)
            for n, similarity, mean_rating in zip(neighbors, similarities, mean_ratings)
            if matrix[n, i] != 0
        ]
        neighbors_ids = [n for n, _, _ in neighbors_rated_i]
        neighbors_similarities = [similarity for _, similarity, _ in neighbors_rated_i]
        neighbors_mean_ratings = [
            mean_rating for _, _, mean_rating in neighbors_rated_i
        ]
        neighbors_ratings_i = [matrix[n, i] for n in neighbors_ids]
        neighbors_ratings_i = [
            (x - y) * z
            for x, y, z in zip(
                neighbors_ratings_i, neighbors_mean_ratings, neighbors_similarities
            )
        ]
        neighbors_ratings_i = numpy.sum(neighbors_ratings_i) / numpy.sum(
            neighbors_similarities
        )

        neighbors_ratings_i += mean_ratings[-1]
        scores[i] = neighbors_ratings_i
    return scores


def get_similarities(matrix: lil_matrix) -> numpy.ndarray:
    return numpy.array(
        [get_similarity_user(matrix=matrix, u=u) for u in range(matrix.shape[0])]
    )


def get_neighbors(matrix: lil_matrix, k: int) -> tuple[numpy.ndarray, numpy.ndarray]:
    neighbors = [
        get_neighbors_user(matrix=matrix, u=u, k=k) for u in range(matrix.shape[0])
    ]
    return numpy.asarray([n for n, _ in neighbors]), numpy.asarray(
        [s for _, s in neighbors]
    )


def get_mean_ratings(matrix: lil_matrix) -> numpy.ndarray:
    return numpy.array(
        [get_mean_ratings_user(matrix=matrix, u=u) for u in range(matrix.shape[0])]
    )


def get_predictions(matrix: lil_matrix, k: int) -> numpy.ndarray:
    neighbors, similarities = get_neighbors(matrix=matrix, k=k)
    mean_ratings = numpy.mean(matrix.toarray(), axis=1)

    scores = numpy.array(matrix.toarray(), dtype=float)
    for u in range(matrix.shape[0]):
        scores_u = scores[u, :]
        for i in range(matrix.shape[1]):
            if scores_u[i] != 0:
                continue

            neighbors_rated_i = [
                (n, similarity, mean_rating)
                for n, similarity, mean_rating in zip(
                    neighbors[u, :], similarities[u, :], mean_ratings
                )
                if matrix[n, i] != 0
            ]

            neighbors_ids = [n for n, _, _ in neighbors_rated_i]
            neighbors_ratings_i = [matrix[n, i] for n in neighbors_ids]
            neighbors_similarities = [
                similarity for _, similarity, _ in neighbors_rated_i
            ]
            neighbors_mean_ratings = [
                mean_rating for _, _, mean_rating in neighbors_rated_i
            ]

            neighbors_ratings_i = [
                (x - y) * z
                for x, y, z in zip(
                    neighbors_ratings_i, neighbors_mean_ratings, neighbors_similarities
                )
            ]
            neighbors_similarity_sum = numpy.sum(neighbors_similarities)
            neighbors_ratings_i = (
                numpy.sum(neighbors_ratings_i) / neighbors_similarity_sum
            )

            neighbors_ratings_i += mean_ratings[u]
            scores_u[i] = neighbors_ratings_i

        scores[u, :] = scores_u
    return scores
