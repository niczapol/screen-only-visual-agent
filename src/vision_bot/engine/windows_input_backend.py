from __future__ import annotations

from vision_bot.core.commands import MouseButton
from vision_bot.movement import InputController, MouseController


class WindowsInputBackend:
    """Thin non-blocking adapter over the existing SendInput controllers."""

    def __init__(
        self,
        keyboard: InputController,
        mouse: MouseController,
    ) -> None:
        self.keyboard = keyboard
        self.mouse = mouse

    def key_down(self, key: str) -> None:
        self.keyboard.press_key(key)

    def key_up(self, key: str) -> None:
        self.keyboard.release_key(key)

    def move_cursor(self, x: int, y: int) -> None:
        self.mouse.move_to(x, y)

    def mouse_down(self, button: MouseButton) -> None:
        self.mouse.button_down(button.value)

    def mouse_up(self, button: MouseButton) -> None:
        self.mouse.button_up(button.value)

    def move_mouse_relative(self, delta_x: int, delta_y: int) -> None:
        self.mouse.move_relative(delta_x, delta_y)
