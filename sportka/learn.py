"""Compatibility imports without import-time downloads or training.

Use sportka_launcher.py for tickets; learn_improved for neural experiments.
"""

from sportka.data import date_to_x, draw, draw_history

__all__ = ["date_to_x", "draw", "draw_history"]


if __name__ == "__main__":
    from sportka.learn_improved import main
    raise SystemExit(main())
