from dataclasses import dataclass, field

import pygame
import pygame_gui

from curvas_omog.settings import POINT_RADIUS, SCREEN_SIZE, UI_INITIAL, UI_SIZE


@dataclass
class Curve:
    points: list[tuple[float, float]]


@dataclass
class EnvironmentState:
    screen: pygame.Surface
    manager: pygame_gui.UIManager
    clock: pygame.Clock
    is_running: bool
    dragging_id: int | None
    active_curve: int
    curves: list[Curve] = field(default_factory=list)

    def add_curve(self, curve: Curve):
        self.curves.append(curve)


def is_click_colliding(
    pos: tuple[float, float],
    points: list[tuple[float, float]],
    ignore_id: int | None = None,
):
    for pos_index, point in enumerate(points):
        if pos_index == ignore_id:
            continue
        distance = ((point[0] - pos[0]) ** 2 + (point[1] - pos[1]) ** 2) ** (1 / 2)
        if distance < 2 * POINT_RADIUS:
            return pos_index
    return None


def handle_event_gui(
    event: pygame.Event,
    state: EnvironmentState,
    radio_group: list[pygame_gui.elements.UIButton],
    add_curve: pygame_gui.elements.UIButton,
):
    if event.type == pygame_gui.UI_BUTTON_PRESSED:
        for i, radio in enumerate(radio_group):
            if event.ui_element == radio:
                state.active_curve = i
        if event.ui_element == add_curve:
            state.curves.append(Curve([]))
            state.active_curve = len(state.curves) - 1
    return state.manager.process_events(event)


def handle_event(
    event: pygame.Event,
    state: EnvironmentState,
    radio_group: list[pygame_gui.elements.UIButton],
    add_curve: pygame_gui.elements.UIButton,
):
    if event.type == pygame.QUIT:
        state.is_running = False
    if handle_event_gui(event, state, radio_group, add_curve):
        return
    pos = pygame.mouse.get_pos()
    if pos[0] <= UI_INITIAL[0] + UI_SIZE[0] and pos[1] <= UI_INITIAL[1] + UI_SIZE[1]:
        return
    if event.type == pygame.MOUSEBUTTONUP:
        if (
            pos_index := is_click_colliding(
                pos, state.curves[state.active_curve].points, state.dragging_id
            )
        ) is None:
            if pos not in state.curves[state.active_curve].points:
                state.curves[state.active_curve].points.append(pos)
        elif state.dragging_id is not None and state.dragging_id != pos_index:
            del state.curves[state.active_curve].points[state.dragging_id]
        state.dragging_id = None
    if event.type == pygame.MOUSEBUTTONDOWN:
        if (
            pos_index := is_click_colliding(
                pos, state.curves[state.active_curve].points
            )
        ) is not None:
            state.dragging_id = pos_index


def get_ui_panel(state: EnvironmentState):
    manager = state.manager
    panel = pygame_gui.elements.UIPanel(
        relative_rect=pygame.Rect(UI_INITIAL, UI_SIZE),
        manager=manager,
    )
    panel.get_container().add_element(
        pygame_gui.elements.UITextBox(
            relative_rect=pygame.Rect((25, 20), (UI_SIZE[0] - 50, 30)),
            html_text=f"Is Dragging: {state.dragging_id is not None}",
            manager=manager,
        )
    )
    radios_initial = 20 + 30 + 20
    height = 30
    padding_bottom = 20
    qtd_curves = len(state.curves)

    radio_curves: list[pygame_gui.elements.UIButton] = list()
    for offset in range(qtd_curves):
        btn_radio = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                (30, radios_initial + ((height + padding_bottom) * offset)),
                (UI_SIZE[0] - 60, 30),
            ),
            text=f"Curve {offset + 1}",
            manager=manager,
        )
        panel.get_container().add_element(btn_radio)
        radio_curves.append(btn_radio)
    btn_add_curve = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(
            (30, radios_initial + ((height + padding_bottom) * qtd_curves)),
            (UI_SIZE[0] - 60, 30),
        ),
        text="Add Curve",
        manager=manager,
    )
    return panel, radio_curves, btn_add_curve


def draw(state: EnvironmentState):
    state.screen.fill("black")

    for point in state.curves[state.active_curve].points:
        pygame.draw.circle(
            state.screen, pygame.color.Color(255, 255, 255), point, POINT_RADIUS
        )

    for point1, point2 in (
        (p1, state.curves[state.active_curve].points[p1_index + 1])
        for p1_index, p1 in enumerate(state.curves[state.active_curve].points[:-1])
    ):
        pygame.draw.line(
            state.screen, pygame.color.Color(255, 255, 255), point1, point2, 1
        )

    state.manager.draw_ui(state.screen)

    pygame.display.flip()


class Environment:
    def __init__(self) -> None:
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()

        screen = pygame.display.set_mode(SCREEN_SIZE)
        manager = pygame_gui.UIManager(SCREEN_SIZE)
        clock = pygame.time.Clock()
        is_running = True

        curves: list[Curve] = [Curve([])]

        dragging_id: int | None = None

        active_curve = 0

        self.state = EnvironmentState(
            screen,
            manager,
            clock,
            is_running,
            dragging_id,
            active_curve,
            curves,
        )

    def main_loop(self):
        while self.state.is_running:
            time_delta = self.state.clock.tick(60) / 1000.0
            _, radio_group, add_curve = get_ui_panel(self.state)
            for event in pygame.event.get():
                handle_event(event, self.state, radio_group, add_curve)

            if self.state.dragging_id is not None:
                pos = pygame.mouse.get_pos()
                if not (
                    pos[0] <= UI_INITIAL[0] + UI_SIZE[0]
                    and pos[1] <= UI_INITIAL[1] + UI_SIZE[1]
                ):
                    self.state.curves[self.state.active_curve].points[
                        self.state.dragging_id
                    ] = pos
            self.state.manager.update(time_delta)
            draw(self.state)

    def quit(self):
        if pygame.get_init():
            pygame.font.quit()
        if pygame.font.get_init():
            pygame.quit()
