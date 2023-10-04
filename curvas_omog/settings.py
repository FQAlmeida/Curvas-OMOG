from typing import Annotated, Literal

import numpy as np
import pygame
from numpy.typing import NDArray

SCREEN_SIZE = (1280, 720)
POINT_RADIUS = 5
UI_INITIAL, UI_SIZE = (0, 0), (300, SCREEN_SIZE[1])
GREY = pygame.color.Color(150, 150, 150, 1)
WHITE = pygame.color.Color(255, 255, 255, 255)
GREEN = pygame.color.Color(0, 255, 0, 255)

K = 4  # Degree of the curve
PointType = Annotated[NDArray[np.float64], Literal["N", 3]]
