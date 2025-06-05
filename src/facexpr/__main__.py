# src/facexpr/__main__.py

import sys
from facexpr.utils.math_helpers import normalize_image_array, compute_mean_and_std

def main():
    """
    A simple default action when someone does `python -m facexpr`.
    If no arguments, it prints usage. Otherwise, it forwards to math_helpers.__main__.
    """
    if len(sys.argv) == 1:
        # No arguments beyond “python -m facexpr”
        print("Usage: python -m facexpr <pixel1> <pixel2> <pixel3> <pixel4>")
        print("Or run: python -m facexpr.math_helpers <args> for the math_helpers demo.")
        sys.exit(0)
    else:
        # Forward everything (except the “-m facexpr” part) to math_helpers.__main__
        # We reassign sys.argv so that math_helpers sees only the pixel values.
        sys.argv = [sys.argv[0]] + sys.argv[1:]
        from facexpr.utils.math_helpers import main as math_main
        math_main()

if __name__ == "__main__":
    main()
