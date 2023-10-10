from dataclasses import dataclass, field
from functools import cache, partial, wraps

import numpy as np
import pygame
from numpy.typing import NDArray

from curvas_omog.settings import (
    GREEN,
    GREY,
    POINT_RADIUS,
    SCREEN_SIZE,
    WHITE,
    K,
    PointType,
)


class Curve:
    points: PointType

    def __init__(self) -> None:
        self.points = np.empty((0, 3))


# def de_boor(k: int, x: int, t, c, p: int):
#     """Evaluates S(x).

#     Arguments
#     ---------
#     k: Index of knot interval that contains x.
#     x: Position.
#     t: Array of knot positions, needs to be padded as described above.
#     c: Array of control points.
#     p: Degree of B-spline.
#     """
#     d = [c[j + k - p] for j in range(0, p + 1)]

#     for r in range(1, p + 1):
#         for j in range(p, r - 1, -1):
#             alpha = (x - t[j + k - p]) / (t[j + 1 + k - r] - t[j + k - p])
#             d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]


#     return d[p]


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

    wrapper.cache_info = cached_wrapper.cache_info # type: ignore
    wrapper.cache_clear = cached_wrapper.cache_clear # type: ignore

    return wrapper


@np_cache
def de_boor(knots: NDArray[np.float64], u: np.float64, degree: np.int64, pi: np.int64):
    if degree == 0:
        if knots[pi] <= u < knots[pi + 1]:
            return np.float64(1)
        return np.float64(0)

    alpha = np.float64(0)
    if not knots[pi + degree] == 0:
        alpha = np.float64(
            ((u - knots[pi]) / (knots[pi + degree]))
            * de_boor(
                knots,
                u,
                degree - 1,
                pi,
            )
        )
    beta = np.float64(0)
    if not knots[pi + degree + 1] - knots[pi + 1] == 0:
        beta = np.float64(
            ((knots[pi + degree + 1] - u) / (knots[pi + degree + 1] - knots[pi + 1]))
            * de_boor(knots, u, degree - 1, pi + 1)
        )

    return alpha + beta


def helper(control_points, knots, degree, u):
    weights = control_points[:, -1]
    points = control_points[:, :-1]
    point = np.array((0.0, 0.0))
    weight = 0
    for pi in range(len(points)):
        d = de_boor(knots, u, degree, np.int64(pi))
        point += weights[pi] * points[pi] * d
        weight += weights[pi] * d
    return point / weight


def evaluate_nurbs_curve(
    control_points: PointType, knots: NDArray[np.float64], num_points=100, degree=K
):
    degree = np.int64(degree)
    u_values = np.linspace(knots[degree], knots[-degree] - 1e-10, num_points)

    curve_points = np.array(
        list(
            map(
                partial(helper, control_points, knots, degree),
                u_values,
            )
        )
    )
    return curve_points


def de_boor1(
    t: float,
    i: int,
    k: int,
    control_points: PointType,
    knot_vector: NDArray[np.float64],
):
    if k == 0:
        return control_points[i, :-1]
    alpha = (t - knot_vector[i]) / (knot_vector[i + k] - knot_vector[i])
    p1 = de_boor1(t, i, k - 1, control_points, knot_vector)
    if alpha == np.Infinity:
        return p1 / control_points[i, -1]
    p2 = de_boor1(t, i + 1, k - 1, control_points, knot_vector)

    # Interpolate with weights
    weighted_interpolated_point = (1 - alpha) * (p1 / control_points[i, -1]) + alpha * (
        p2 / control_points[i + 1, -1]
    )
    return weighted_interpolated_point


def boor(t, degree, i, knots):
    if degree == 0:
        return 1.0 if knots[i] <= t < knots[i + 1] else 0.0

    if knots[i + degree] == knots[i]:
        c1 = 0.0
    else:
        c1 = (
            (t - knots[i])
            / (knots[i + degree] - knots[i])
            * boor(t, degree - 1, i, knots)
        )

    if knots[i + degree + 1] == knots[i + 1]:
        c2 = 0.0
    else:
        c2 = (
            (knots[i + degree + 1] - t)
            / (knots[i + degree + 1] - knots[i + 1])
            * boor(t, degree - 1, i + 1, knots)
        )

    return c1 + c2


@dataclass
class EnvironmentState:
    screen: pygame.Surface
    clock: pygame.time.Clock
    font: pygame.font.Font
    is_running: bool
    dragging_id: int | None

    active_curve_index: int
    curves: list[Curve] = field(default_factory=list)

    should_delete_on_collide = False
    should_snap = True
    should_draw_support_lines = True
    should_draw_support_points = True

    def add_curve(self, curve: Curve):
        self.curves.append(curve)

    @property
    def active_curve(self):
        return self.curves[self.active_curve_index]


def is_click_colliding(
    pos: tuple[float, float],
    points: PointType,
    ignore_id: int | None = None,
) -> int | None:
    if not points.any():
        return None
    if ignore_id is not None:
        points = points[np.arange(len(points)) != ignore_id]
    np_pos = np.array(
        [
            pos,
        ]
    )
    distances = np.array(
        np.sqrt(np.sum(np.square(np.subtract(np_pos, points[:, :-1])), axis=1))
    )
    if len(indexes := np.argwhere(distances < 2 * POINT_RADIUS)) > 0:
        rindex = indexes[0, 0]
        return rindex + (0 if ignore_id is None else 0 if ignore_id > rindex else 1)
    return None


def handle_event(
    event: pygame.event.Event,
    state: EnvironmentState,
):
    if event.type == pygame.QUIT:
        state.is_running = False
    pos = pygame.mouse.get_pos()
    if event.type == pygame.MOUSEBUTTONUP:
        handle_mouse_btn_up(state, pos)
    if event.type == pygame.MOUSEBUTTONDOWN:
        handle_mouse_btn_down(state, pos)
    if event.type == pygame.KEYUP:
        handle_keybd_up(event, state)


def handle_keybd_up(event: pygame.event.Event, state: EnvironmentState):
    match event.key:
        case pygame.K_KP1:
            state.should_delete_on_collide = not state.should_delete_on_collide
        case pygame.K_KP3:
            state.should_snap = not state.should_snap
        case pygame.K_KP4:
            state.should_draw_support_lines = not state.should_draw_support_lines
        case pygame.K_KP5:
            state.should_draw_support_points = not state.should_draw_support_points
        case pygame.K_KP6:
            state.curves = [Curve()]
            state.active_curve_index = 0
            rng = np.random.default_rng(1337)
            state.active_curve.points = rng.random((10000, 3)) * (*SCREEN_SIZE, 1)
        case pygame.K_KP2:
            state.curves = [Curve()]
            state.active_curve_index = 0


def handle_mouse_btn_up(state: EnvironmentState, pos: tuple[int, int]):
    pos_point = np.array((*pos, 1))
    if (
        pos_index := is_click_colliding(
            pos, state.curves[state.active_curve_index].points, state.dragging_id
        )
    ) is None:
        if not np.isin(
            state.curves[state.active_curve_index].points[:, :-1], pos_point
        ).any():
            state.curves[state.active_curve_index].points = np.concatenate(
                (state.curves[state.active_curve_index].points, [pos_point]),
                axis=0,
            )
    elif state.dragging_id is not None and state.dragging_id != pos_index:
        if state.should_delete_on_collide:
            state.curves[state.active_curve_index].points = state.curves[
                state.active_curve_index
            ].points[
                np.arange(len(state.curves[state.active_curve_index].points))
                != state.dragging_id
            ]
        elif state.should_snap:
            state.active_curve.points[state.dragging_id] = state.active_curve.points[
                pos_index
            ]

    state.dragging_id = None


def handle_mouse_btn_down(state: EnvironmentState, pos: tuple[int, int]):
    if (
        pos_index := is_click_colliding(
            pos, state.curves[state.active_curve_index].points
        )
    ) is not None:
        state.dragging_id = pos_index


def draw_texts(state: EnvironmentState):
    should_delete_on_collide_text = state.font.render(
        f"Should Delete: {state.should_delete_on_collide}",
        True,
        "green" if state.should_delete_on_collide else "red",
    )
    state.screen.blit(should_delete_on_collide_text, (20, 20))
    active_curve_text = state.font.render(
        f"Active Curve: {state.active_curve_index+1}",
        True,
        "green",
    )
    state.screen.blit(active_curve_text, (20, 40))
    qtd_points_text = state.font.render(
        f"Qtd Points: {len(state.active_curve.points)}",
        True,
        "green",
    )
    state.screen.blit(qtd_points_text, (20, 60))
    should_snap_text = state.font.render(
        f"Should Snap in Place: {state.should_snap}",
        True,
        "green" if state.should_snap else "red",
    )
    state.screen.blit(should_snap_text, (20, 80))
    should_draw_support_text = state.font.render(
        f"Should Draw Support Lines: {state.should_draw_support_lines}",
        True,
        "green" if state.should_draw_support_lines else "red",
    )
    state.screen.blit(should_draw_support_text, (20, 100))
    should_draw_support_points_text = state.font.render(
        f"Should Draw Support Points: {state.should_draw_support_points}",
        True,
        "green" if state.should_draw_support_points else "red",
    )
    state.screen.blit(should_draw_support_points_text, (20, 120))


def draw(state: EnvironmentState):
    state.screen.fill("black")

    if state.should_draw_support_points:
        for point in state.active_curve.points:
            pygame.draw.circle(state.screen, GREEN, point[:-1], POINT_RADIUS)

    if state.should_draw_support_lines:
        for point1, point2 in (
            (p1, state.active_curve.points[p1_index + 1])
            for p1_index, p1 in enumerate(state.active_curve.points[:-1])
        ):
            pygame.draw.line(state.screen, GREY, point1[:-1], point2[:-1], 1)

    n = len(state.active_curve.points)
    if n > K:
        knot_vector = np.array(
            [0] * K + list(range(n - K+1)) + [n - K] * K, dtype="int"
        ) / (n - K)
        curve_points = evaluate_nurbs_curve(
            state.active_curve.points, knot_vector, 50 * n
        )

        for point1, point2 in (
            (p1, curve_points[p1_index + 1])
            for p1_index, p1 in enumerate(curve_points[:-1])
        ):
            pygame.draw.line(state.screen, WHITE, point1, point2, 1)
        # qtd_points_text = state.font.render(
        #     f"Qtd Points Knot: {len(knot_vector)}",
        #     True,
        #     "green",
        # )
        # state.screen.blit(qtd_points_text, (100, 20))
    draw_texts(state)
    pygame.display.flip()


# Create a knot vector


class Environment:
    def __init__(self) -> None:
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()

        # print(sorted(pygame.font.get_fonts()))

        font = pygame.font.SysFont(["sans", pygame.font.get_default_font()], 16)
        screen = pygame.display.set_mode(SCREEN_SIZE, pygame.SCALED)
        clock = pygame.time.Clock()
        is_running = True

        curves: list[Curve] = [Curve()]

        dragging_id: int | None = None

        active_curve = 0

        self.state = EnvironmentState(
            screen,
            clock,
            font,
            is_running,
            dragging_id,
            active_curve,
            curves,
        )

    def main_loop(self):
        while self.state.is_running:
            for event in pygame.event.get():
                handle_event(event, self.state)

            if self.state.dragging_id is not None:
                pos = pygame.mouse.get_pos()
                self.state.curves[self.state.active_curve_index].points[
                    self.state.dragging_id
                ] = np.array((*pos, 1))
            draw(self.state)
            self.state.clock.tick(60)

    def quit(self):
        if pygame.get_init():
            pygame.font.quit()
        if pygame.font.get_init():
            pygame.quit()
