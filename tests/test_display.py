import unittest

import mlx.core as mx

from fastmlx.display import BatchDisplay, GridDisplay


class TestDisplayUtilities(unittest.TestCase):
    def test_batch_display_prepare(self) -> None:
        images = mx.zeros((2, 4, 4, 1))
        disp = BatchDisplay(image=images, title="x")
        fig = disp.prepare()
        self.assertTrue(hasattr(fig, "to_html"))

    def test_grid_display_prepare(self) -> None:
        images = mx.zeros((2, 4, 4, 1))
        col1 = BatchDisplay(image=images, title="a")
        col2 = BatchDisplay(image=images, title="b")
        grid = GridDisplay([col1, col2])
        fig = grid.prepare()
        self.assertTrue(hasattr(fig, "to_html"))


if __name__ == "__main__":
    unittest.main()
