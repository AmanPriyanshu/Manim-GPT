from manim import *
import numpy as np

config.pixel_height = 720
config.pixel_width = 1280
config.frame_height = 8.0
config.frame_width = 14.0

class ConvVisualization(Scene):
    def construct(self):
        # Create a 2D grid to represent the input image
        grid = VGroup(*[
            VGroup(*[
                Square(side_length=0.5, stroke_color=GREY, fill_color=WHITE, fill_opacity=0.5)
                for _ in range(8)
            ]).arrange(RIGHT, buff=0)
            for _ in range(8)
        ]).arrange(DOWN, buff=0)
        self.add(grid)

        # Highlight a section of the grid to show it's being processed
        highlight_rect = Rectangle(
            width=1.5, height=1.5, 
            stroke_color=YELLOW, fill_color=YELLOW, fill_opacity=0.5
        )
        highlight_rect.move_to(grid[3][3].get_center())

        self.play(Create(highlight_rect))
        self.wait(1)

class ConvolutionalFilter(Scene):
    def construct(self):
        # Image Grid
        grid = VGroup()
        image_array = np.random.randint(0, 256, (5, 5)) # Example 5x5 image
        for i in range(image_array.shape[0]):
            for j in range(image_array.shape[1]):
                square = Square(side_length=0.5)
                square.move_to(np.array([j - 2, 2 - i, 0]))
                square.set_fill(GRAY, opacity=image_array[i, j] / 256)
                grid.add(square)

        grid.shift(LEFT * 2)
        self.add(grid)

        # Define the filter
        filter_size = 3
        filter_highlights = VGroup()

        for i in range(filter_size):
            for j in range(filter_size):
                filter_highlight = Square(side_length=0.5)
                filter_highlight.move_to(np.array([j - 1, 1 - i, 0]))
                filter_highlight.set_color(YELLOW)
                filter_highlight.set_stroke(width=4)
                filter_highlights.add(filter_highlight)

        def update_filter_highlights(filter_highlights, time):
            y, x = divmod(int(time), 3)
            filter_position = np.array([x - 1, 1 - y, 0])
            filter_highlights.move_to(filter_position + LEFT * 2)

        filter_highlights.add_updater(update_filter_highlights)
        self.add(filter_highlights)

        # Slide the filter across the image
        self.play(UpdateFromAlphaFunc(filter_highlights, lambda m, a: update_filter_highlights(m, a * 9)), run_time=5)
        self.wait()

class CNNFeatureMap(Scene):
    def construct(self):
        # Create input feature map matrix
        input_matrix = IntegerMatrix(np.random.randint(0, 10, (5, 5)))
        input_matrix.shift(LEFT * 3)

        # Create filter matrices
        filter1 = IntegerMatrix(np.random.randint(-1, 2, (3, 3)), v_buff=0.7)
        filter1.next_to(input_matrix, RIGHT, buff=2)

        filter2 = IntegerMatrix(np.random.randint(-1, 2, (3, 3)), v_buff=0.7)
        filter2.next_to(filter1, RIGHT, buff=1)

        # Create placeholder for resulting feature map
        result_matrix = IntegerMatrix(np.zeros((3, 3)), v_buff=0.7, h_buff=1.4)
        result_matrix.to_edge(RIGHT)

        # Displaying elements
        self.play(Create(input_matrix))
        self.play(Create(filter1), Create(filter2))
        self.play(TransformFromCopy(filter1, result_matrix))
        
        # Animate feature map computation
        self.wait()
        for i in range(3):
            for j in range(3):
                self.play(
                    result_matrix.get_entries()[i * 3 + j].animate.set_value(
                        np.sum(
                            input_matrix.get_entries()[i:i+3, j:j+3].flatten() * filter1.get_entries() +
                            input_matrix.get_entries()[i:i+3, j:j+3].flatten() * filter2.get_entries()
                        )
                    )
                )
        
        self.wait(2)