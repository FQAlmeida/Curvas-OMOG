from typing import Annotated, Literal

import numpy as np
import pygame
from numpy.typing import NDArray
from screeninfo import get_monitors

primary_monitor = next(iter(filter(lambda mon: mon.is_primary, get_monitors())))

SCREEN_SIZE = tuple(
    map(lambda res: res * 0.85, (primary_monitor.width, primary_monitor.height))
)
POINT_RADIUS = 5
UI_INITIAL, UI_SIZE = (0, 0), (300, SCREEN_SIZE[1])
GREY = pygame.color.Color(150, 150, 150, 1)
WHITE = pygame.color.Color(255, 255, 255, 255)
GREEN = pygame.color.Color(0, 255, 0, 255)
RED = pygame.color.Color(255, 0, 0, 255)
BLUE = pygame.color.Color(0, 0, 255, 255)

K = 4  # Degree of the curve
PointType = Annotated[NDArray[np.float64], Literal["N", 3]]
