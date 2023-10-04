from dataclasses import dataclass, field

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
def de_boor(
    t: float,
    i: int,
    k: int,
    control_points: PointType,
    knot_vector: NDArray[np.float64],
):
    if k == 0:
        return control_points[i, :-1]
    alpha = (t - knot_vector[i]) / (knot_vector[i + k] - knot_vector[i])
    if alpha == np.Infinity:
        print(alpha, t, i, k)
    p1 = de_boor(t, i, k - 1, control_points, knot_vector)
    p2 = de_boor(t, i + 1, k - 1, control_points, knot_vector)

    # Interpolate with weights
    weighted_interpolated_point = (1 - alpha) * (p1 / control_points[i, -1]) + alpha * (
        p2 / control_points[i + 1, -1]
    )
    return weighted_interpolated_point


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

    def add_curve(self, curve: Curve):
        self.curves.append(curve)

    @property
    def active_curve(self):
        return self.curves[self.active_curve_index]


def is_click_colliding(
    pos: tuple[float, float],
    points: PointType,
    ignore_id: int | None = None,
):
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
        return indexes[0]
    return None


def handle_event(
    event: pygame.event.Event,
    state: EnvironmentState,
):
    if event.type == pygame.QUIT:
        state.is_running = False
    pos = pygame.mouse.get_pos()
    pos_point = np.array((*pos, 1))
    if event.type == pygame.MOUSEBUTTONUP:
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
        elif (
            state.dragging_id is not None
            and state.dragging_id != pos_index
            and state.should_delete_on_collide
        ):
            state.curves[state.active_curve_index].points = state.curves[
                state.active_curve_index
            ].points[
                np.arange(len(state.curves[state.active_curve_index].points))
                != state.dragging_id
            ]
        state.dragging_id = None
    if event.type == pygame.MOUSEBUTTONDOWN:
        if (
            pos_index := is_click_colliding(
                pos, state.curves[state.active_curve_index].points
            )
        ) is not None:
            state.dragging_id = pos_index
    if event.type == pygame.KEYUP and event.key == pygame.K_KP1:
        state.should_delete_on_collide = not state.should_delete_on_collide


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


def draw(state: EnvironmentState):
    state.screen.fill("black")

    for point in state.active_curve.points:
        pygame.draw.circle(state.screen, GREEN, point[:-1], POINT_RADIUS)

    for point1, point2 in (
        (p1, state.active_curve.points[p1_index + 1])
        for p1_index, p1 in enumerate(state.active_curve.points[:-1])
    ):
        pygame.draw.line(state.screen, GREY, point1[:-1], point2[:-1], 1)

    curve_points = []
    n = len(state.active_curve.points)
    if K + 3 < n:
        knot_vector = knot_vector = np.concatenate(
            (
                np.zeros(K),  # Clamped start
                np.linspace(1, n - K, n - K),  # Interior knots
                np.full(shape=K, fill_value=K - 1),  # Clamped end
            )
        )
        print(knot_vector)
        t_step = 0.01
        for t in np.arange(K - 1, n + 1, t_step):
            curve_point = de_boor(t, 0, K, state.active_curve.points, knot_vector)
            curve_points.append(curve_point)
        qtd_points_text = state.font.render(
            f"Qtd Points Knot: {len(knot_vector)}",
            True,
            "green",
        )
        state.screen.blit(qtd_points_text, (20, 80))
    print(curve_points)
    for point1, point2 in (
        (p1, curve_points[p1_index + 1])
        for p1_index, p1 in enumerate(curve_points[:-1])
    ):
        pygame.draw.line(state.screen, WHITE, point1, point2, 1)
    draw_texts(state)
    pygame.display.flip()


# Create a knot vector


class Environment:
    def __init__(self) -> None:
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()

        font = pygame.font.SysFont(["comic sans"], 16)
        screen = pygame.display.set_mode(SCREEN_SIZE)
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
