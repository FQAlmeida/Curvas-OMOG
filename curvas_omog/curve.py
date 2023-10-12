from abc import ABC, abstractmethod
from functools import cache, partial, wraps

import numpy as np
from numpy.typing import NDArray
from scipy.special import comb

from curvas_omog.settings import K, PointType


def np_cache(function):
    @cache
    def cached_wrapper(*args, **kwargs):
        args = [np.array(a) if isinstance(a, tuple) else a for a in args]
        kwargs = {
            k: np.array(v) if isinstance(v, tuple) else v for k, v in kwargs.items()
        }

        return function(*args, **kwargs)

    @wraps(function)
    def wrapper(*args, **kwargs):
        args = [tuple(a) if isinstance(a, np.ndarray) else a for a in args]
        kwargs = {
            k: tuple(v) if isinstance(v, np.ndarray) else v for k, v in kwargs.items()
        }
        return cached_wrapper(*args, **kwargs)

    wrapper.cache_info = cached_wrapper.cache_info  # type: ignore[reportGeneralTypeIssues]
    wrapper.cache_clear = cached_wrapper.cache_clear  # type: ignore[reportGeneralTypeIssues]

    return wrapper


class Curve(ABC):
    points: PointType

    def __init__(self, degree=K) -> None:
        self.points = np.empty((0, 3), dtype=np.float64)
        self.degree = np.int64(degree)

    @abstractmethod
    def evaluate_curve(self, num_points=100) -> NDArray[np.float64]:
        ...


class Nurb(Curve):
    @np_cache
    def de_boor(
        self, knots: NDArray[np.float64], u: np.float64, degree: np.int64, pi: np.int64,
    ):
        if degree == 0:
            if knots[pi] <= u < knots[pi + 1]:
                return np.float64(1)
            return np.float64(0)

        alpha = np.float64(0)
        if knots[pi + degree] - knots[pi] != 0:
            alpha = np.float64(
                ((u - knots[pi]) / (knots[pi + degree] - knots[pi]))
                * self.de_boor(
                    knots,
                    u,
                    degree - 1,
                    pi,
                ),
            )
        beta = np.float64(0)
        if knots[pi + degree + 1] - knots[pi + 1] != 0:
            beta = np.float64(
                (
                    (knots[pi + degree + 1] - u)
                    / (knots[pi + degree + 1] - knots[pi + 1])
                )
                * self.de_boor(knots, u, degree - 1, pi + 1),
            )

        return alpha + beta

    def helper(
        self,
        knots: NDArray[np.float64],
        degree: np.int64,
        u: np.float64,
    ):
        weights = self.points[:, -1]
        points = self.points[:, :-1]
        point = np.array((0.0, 0.0))
        weight = 0
        for pi in range(len(points)):
            d = self.de_boor(knots, u, degree, np.int64(pi))
            point += weights[pi] * points[pi] * d
            weight += weights[pi] * d
        if weight in {np.inf, 0}:
            return self.points[-1, :-1]
        return point / weight

    def evaluate_curve(
        self,
        num_points=100,
    ):
        n = len(self.points)
        degree = max(min(n-1, int(self.degree)), 0)
        knots = np.array(
            [0] * degree + list(range(n - degree + 1)) + [n - degree] * degree, dtype="int",
        ) / (n - degree)
        degree = np.int64(degree)
        u_values = np.linspace(knots[degree], knots[-degree], num_points)

        return np.array(
            list(
                map(
                    partial(self.helper, knots, degree),
                    u_values,
                ),
            ),
        )


class Bezier(Curve):
    @np_cache
    def bernstein(self, u: np.float64, n: np.int64, pi: np.int64):
        c = comb(n, pi, exact=False)
        a1 = u**pi
        a2 = (1 - u) ** (n - pi)
        return c * a1 * a2

    def helper(
        self,
        n: np.int64,
        u: np.float64,
    ):
        points = self.points[:, :-1]
        point = np.array((0.0, 0.0))
        for pi in range(len(points)):
            d = self.bernstein(u, n, np.int64(pi))
            point += points[pi] * d
        return point

    def evaluate_curve(self, num_points=100):
        n = len(self.points)
        u_values = np.linspace(0, 1, num_points)

        return np.array(
            list(
                map(
                    partial(self.helper, np.int64(n - 1)),
                    u_values,
                ),
            ),
        )
