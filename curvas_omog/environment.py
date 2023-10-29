from dataclasses import dataclass, field

import numpy as np
import pygame

from curvas_omog.curve import Bezier, Curve, Nurb
from curvas_omog.settings import (
    BLUE,
    GREEN,
    GREY,
    POINT_RADIUS,
    RED,
    SCREEN_SIZE,
    WHITE,
    PointType,
)


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
        ],
    )
    distances = np.array(
        np.sqrt(np.sum(np.square(np.subtract(np_pos, points[:, :-1])), axis=1)),
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
    if event.type == pygame.MOUSEBUTTONUP and event.button == pygame.BUTTON_LEFT:
        handle_mouse_btn_up(state, pos)
    if event.type == pygame.MOUSEBUTTONDOWN and event.button == pygame.BUTTON_LEFT:
        handle_mouse_btn_down(state, pos)
    if event.type == pygame.MOUSEWHEEL:
        handle_mouse_wheel(event, state, pos)
    if event.type == pygame.KEYUP:
        handle_keybd_up(event, state)


def handle_mouse_wheel(
    event: pygame.event.Event,
    state: EnvironmentState,
    pos: tuple[int, int],
):
    if (
        pos_index := is_click_colliding(
            pos,
            state.active_curve.points,
            state.dragging_id,
        )
    ) is not None:
        state.active_curve.points[pos_index][-1] += 0.1 * event.y


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
            state.curves = [Nurb()]
            state.active_curve_index = 0
            rng = np.random.default_rng(1337)
            state.active_curve.points = rng.random((10, 3)) * (*SCREEN_SIZE, 1)
        case pygame.K_KP7:
            state.active_curve_index -= 1
            if state.active_curve_index < 0:
                state.active_curve_index = 0
        case pygame.K_KP8:
            state.active_curve_index += 1
            if state.active_curve_index >= len(state.curves):
                state.active_curve_index = len(state.curves) - 1
        case pygame.K_KP9:
            state.add_curve(Nurb())
            state.active_curve_index = len(state.curves) - 1
        case pygame.K_KP_PLUS:
            state.add_curve(Bezier())
            state.active_curve_index = len(state.curves) - 1
        case pygame.K_KP_DIVIDE:
            handle_c_0(state)
        case pygame.K_p:
            handle_g_1(state)
        case pygame.K_o:
            handle_g_2(state)
        case pygame.K_KP2:
            state.curves = [Nurb()]
            state.active_curve_index = 0


def handle_mouse_btn_up(state: EnvironmentState, pos: tuple[int, int]):
    pos_point = np.array((*pos, 1))
    if (
        pos_index := is_click_colliding(
            pos,
            state.active_curve.points,
            state.dragging_id,
        )
    ) is None:
        if not np.isin(state.active_curve.points[:, :-1], pos_point).any():
            state.active_curve.points = np.concatenate(
                (state.active_curve.points, [pos_point]),
                axis=0,
            )
    elif state.dragging_id is not None and state.dragging_id != pos_index:
        if state.should_delete_on_collide:
            state.active_curve.points = state.curves[state.active_curve_index].points[
                np.arange(len(state.active_curve.points)) != state.dragging_id
            ]
        elif state.should_snap:
            state.active_curve.points[state.dragging_id] = state.active_curve.points[
                pos_index
            ]

    state.dragging_id = None


def handle_mouse_btn_down(state: EnvironmentState, pos: tuple[int, int]):
    if (pos_index := is_click_colliding(pos, state.active_curve.points)) is not None:
        state.dragging_id = pos_index


def handle_c_0(state: EnvironmentState):
    for curva_1, curva_2, curva_2_index in (
        (curva_1, state.curves[curva_1_index + 1], curva_1_index + 1)
        for curva_1_index, curva_1 in filter(
            lambda x: len(x[1].points) != 0 and len(state.curves[x[0] + 1].points) != 0,
            enumerate(state.curves[:-1]),
        )
    ):
        end_point = curva_1.points[-1]
        start_point = curva_2.points[0]
        translation = np.append(start_point[:-1] - end_point[:-1], 0)
        state.curves[curva_2_index].points = (
            state.curves[curva_2_index].points - translation
        )


def handle_g_1(state: EnvironmentState):
    handle_c_0(state)
    for curva_1, curva_2, curva_2_index in (
        (curva_1, state.curves[curva_1_index + 1], curva_1_index + 1)
        for curva_1_index, curva_1 in filter(
            lambda x: len(x[1].points) != 0 and len(state.curves[x[0] + 1].points) != 0,
            enumerate(state.curves[:-1]),
        )
    ):
        end_point, end_point_before = curva_1.points[-1, :-1], curva_1.points[-2, :-1]
        start_point, start_point_after = curva_2.points[0, :-1], curva_2.points[1, :-1]

        direction = end_point - end_point_before
        direction_mod = np.linalg.norm(direction)
        direction /= direction_mod

        position_dir = start_point_after - start_point
        position_mag = np.linalg.norm(position_dir)

        position = np.append(start_point + (direction * position_mag), 1)

        state.curves[curva_2_index].points[1] = position


def handle_g_2(state: EnvironmentState):
    handle_g_1(state)
    for curva_1, curva_2, curva_2_index in (
        (curva_1, state.curves[curva_1_index + 1], curva_1_index + 1)
        for curva_1_index, curva_1 in filter(
            lambda x: len(x[1].points) != 0 and len(state.curves[x[0] + 1].points) != 0,
            enumerate(state.curves[:-1]),
        )
    ):
        end_point, end_point_before, end_point_before_before = (
            curva_1.points[-1, :-1],
            curva_1.points[-2, :-1],
            curva_1.points[-3, :-1],
        )
        start_point, start_point_after, start_point_after_after = (
            curva_2.points[0, :-1],
            curva_2.points[1, :-1],
            curva_2.points[2, :-1],
        )

        a_2 = start_point - start_point_after
        a_2 /= np.linalg.norm(a_2)
        b_2 = start_point_after_after - start_point_after
        b_2_mag = np.linalg.norm(b_2)

        a_1 = end_point - end_point_before
        a_1 /= np.linalg.norm(a_1)
        b_1 = end_point_before_before - end_point_before
        b_1 /= np.linalg.norm(b_1)

        angle = np.arccos(np.dot(b_1, a_1))
        if all(
            (
                a_1[1] >= 0,
                b_1[0] <= 0,
            ),
        ) or all(
            (
                a_1[1] <= 0,
                b_1[0] >= 0,
            ),
        ):
            angle += np.pi

        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        )
        new_b_2 = np.dot(rotation_matrix, a_2)

        new_b_2_mag = new_b_2 * b_2_mag
        new_point = start_point_after + new_b_2_mag

        state.curves[curva_2_index].points[2] = np.append(new_point, 1)
        handle_g_1(state)


def draw_texts(state: EnvironmentState):
    should_delete_on_collide_text = state.font.render(
        f"Should Delete: {state.should_delete_on_collide}",
        True,  # noqa: FBT003
        "green" if state.should_delete_on_collide else "red",
    )
    state.screen.blit(should_delete_on_collide_text, (20, 20))
    active_curve_text = state.font.render(
        f"Active Curve: {state.active_curve_index+1}",
        True,  # noqa: FBT003
        "green",
    )
    state.screen.blit(active_curve_text, (20, 40))
    qtd_points_text = state.font.render(
        f"Qtd Points: {len(state.active_curve.points)}",
        True,  # noqa: FBT003
        "green",
    )
    state.screen.blit(qtd_points_text, (20, 60))
    should_snap_text = state.font.render(
        f"Should Snap in Place: {state.should_snap}",
        True,  # noqa: FBT003
        "green" if state.should_snap else "red",
    )
    state.screen.blit(should_snap_text, (20, 80))
    should_draw_support_text = state.font.render(
        f"Should Draw Support Lines: {state.should_draw_support_lines}",
        True,  # noqa: FBT003
        "green" if state.should_draw_support_lines else "red",
    )
    state.screen.blit(should_draw_support_text, (20, 100))
    should_draw_support_points_text = state.font.render(
        f"Should Draw Support Points: {state.should_draw_support_points}",
        True,  # noqa: FBT003
        "green" if state.should_draw_support_points else "red",
    )
    state.screen.blit(should_draw_support_points_text, (20, 120))
    qtd_curves_text = state.font.render(
        f"Qtd Curves: {len(state.curves)}",
        True,  # noqa: FBT003
        "green",
    )
    state.screen.blit(qtd_curves_text, (20, 140))


def draw(state: EnvironmentState):
    state.screen.fill("black")

    if state.should_draw_support_points:
        for point in state.active_curve.points:
            pygame.draw.circle(
                state.screen,
                BLUE if point[-1] == 0 else GREEN if point[-1] >= 0 else RED,
                point[:-1],
                POINT_RADIUS,
            )

    if state.should_draw_support_lines:
        for point1, point2 in (
            (p1, state.active_curve.points[p1_index + 1])
            for p1_index, p1 in enumerate(state.active_curve.points[:-1])
        ):
            pygame.draw.line(state.screen, GREY, point1[:-1], point2[:-1], 1)

    for curve in state.curves:
        n = len(curve.points)
        curve_points = curve.evaluate_curve(50 * n)
        for point1, point2 in (
            (p1, curve_points[p1_index + 1])
            for p1_index, p1 in enumerate(curve_points[:-1])
        ):
            pygame.draw.line(state.screen, WHITE, point1, point2, 1)
    draw_texts(state)
    pygame.display.flip()


class Environment:
    def __init__(self) -> None:
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()

        font = pygame.font.SysFont(["sans", pygame.font.get_default_font()], 16)
        screen = pygame.display.set_mode(SCREEN_SIZE, pygame.SCALED)
        clock = pygame.time.Clock()
        is_running = True

        curves: list[Curve] = [Nurb()]

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
                self.state.active_curve.points[self.state.dragging_id] = np.array(
                    (*pos, self.state.active_curve.points[self.state.dragging_id, -1]),
                )
            draw(self.state)
            self.state.clock.tick(60)

    def quit_env(self):
        if pygame.get_init():
            pygame.font.quit()
        if pygame.font.get_init():
            pygame.quit()
