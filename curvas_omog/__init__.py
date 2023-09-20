import pygame

pygame.init()
pygame.font.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True

points: list[tuple[float, float]] = list()

dragging_id: int | None = None
POINT_RADIUS = 5


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


while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONUP:
            pos = pygame.mouse.get_pos()
            if (pos_index := is_click_colliding(pos, points, dragging_id)) is None:
                if pos not in points:
                    points.append(pos)
            elif dragging_id is not None and dragging_id != pos_index:
                del points[dragging_id]
            dragging_id = None
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            if (pos_index := is_click_colliding(pos, points)) is not None:
                dragging_id = pos_index
    screen.fill("black")

    if dragging_id is not None:
        points[dragging_id] = pygame.mouse.get_pos()

    for point in points:
        pygame.draw.circle(
            screen, pygame.color.Color(255, 255, 255), point, POINT_RADIUS
        )
    font = pygame.font.SysFont("sans", 16)
    img = font.render(
        f"Is Dragging: {dragging_id if dragging_id is not None else False}", True, "red"
    )
    screen.blit(img, (20, 20))
    pygame.display.flip()

    clock.tick(60)  # limits FPS to 60

pygame.font.quit()
pygame.quit()
