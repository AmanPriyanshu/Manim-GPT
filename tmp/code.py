from manim import *

class Simple2DGrid(Scene):
    def construct(self):
        # Create a grid to represent a 2D image
        grid = VGroup()
        for x in range(5):  # 5x5 grid
            for y in range(5):
                square = Square(side_length=0.4, fill_color=BLUE, fill_opacity=0.4)
                square.move_to(x * RIGHT + y * UP)
                grid.add(square)
        grid.move_to(ORIGIN)
        
        # Highlight the grid
        input_layer_label = Text("Input Image Layer", font_size=36)
        input_layer_label.next_to(grid, UP, buff=0.5)

        # Add grid and label to scene
        self.play(Create(grid))
        self.play(Write(input_layer_label))
        
        # Keep the image on screen for a bit
        self.wait(2)

class ConvolutionalLayer(Scene):
    def construct(self):
        # Create a grid representing the input image
        input_grid = VGroup(*[Square() for _ in range(16)])
        input_grid.arrange_in_grid(rows=4, buff=0)
        input_grid.set_fill(BLUE, opacity=0.4)
        self.play(Create(input_grid))

        # Show filters moving across the input image
        filters_group = VGroup(
            Square(side_length=0.9, color=RED),
            Square(side_length=0.9, color=GREEN),
            Square(side_length=0.9, color=YELLOW)
        )
        filters_group.arrange(buff=0.1).move_to(input_grid[0])

        for filter in filters_group:
            self.play(filter.animate.shift(RIGHT*0.9*3), run_time=2)
            self.play(filter.animate.shift(DOWN*0.9*3), run_time=2)

        # Overlay the input grid on top of the movement to illustrate convolution
        self.play(input_grid.animate.set_opacity(0.2))

        # Adding feature map result
        feature_map_result = Text("Feature Map Result", color=WHITE).next_to(input_grid, DOWN)
        self.play(Write(feature_map_result))

import numpy as np

class ReLUActivation(Scene):
    def construct(self):
        # Creating a grid to mimic convolutional layer's feature maps
        grid = VGroup(*[VGroup(*[Square(side_length=0.5) for _ in range(5)]).arrange(RIGHT) for _ in range(5)]).arrange(DOWN)

        # Assigning dummy data to the grid (some values above zero, some not)
        values = np.random.uniform(-1, 1, (5, 5))

        # Adding values to the grid
        for i, row in enumerate(grid):
            for j, square in enumerate(row):
                color = YELLOW if values[i][j] > 0 else GRAY  # Highlight active neurons
                text = Text(f"{values[i][j]:.2f}", font_size=24).move_to(square.get_center())
                square.set_fill(color, opacity=0.5)
                square.add(text)

        self.play(Create(grid))
        self.wait()

        # Animation for ReLU activation
        animations = []
        for i, row in enumerate(values):
            for j, val in enumerate(row):
                if val <= 0:
                    animations.append(FadeOut(grid[i][j]))
                else:
                    grid[i][j].set_fill(YELLOW, opacity=0.8)

        self.play(*animations)
        self.wait(2)

class MaxPooling(Scene):
    def construct(self):
        # Original feature map (3x3)
        original_matrix = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])

        # Visual representation of original feature map
        matrix_mobject = IntegerMatrix(original_matrix, h_buff=2, v_buff=2)
        matrix_mobject.scale(0.5)

        # Title for the original map
        orig_title = Text("Original Feature Map").next_to(matrix_mobject, UP)

        # Axes to align matrices correctly
        axes = Axes(x_range=[0, 3, 1], y_range=[0, 3, 1],
                    height=2, width=2, axis_config={"stroke_opacity": 0})
        axes.move_to(matrix_mobject)

        # Simulate max pooling operation (2x2) with stride 2 
        pooled_matrix = np.array([
            [5, 6],
            [8, 9]
        ])

        # Visual representation of the pooled feature map
        pooled_matrix_mobject = IntegerMatrix(pooled_matrix, h_buff=2, v_buff=2, element_to_mobject_config={"fill_color": YELLOW})
        pooled_matrix_mobject.scale(0.5)
        pooled_matrix_mobject.next_to(matrix_mobject, RIGHT, buff=1)

        # Title for the pooled map
        pooled_title = Text("Pooled Feature Map").next_to(pooled_matrix_mobject, UP)

        # Animations
        self.play(Create(matrix_mobject), Write(orig_title))
        self.wait(1)
        self.play(ReplacementTransform(matrix_mobject.copy(), pooled_matrix_mobject), Write(pooled_title))
        self.wait(1)

class CNNVisualization(Scene):
    def construct(self):
        # Initialize the CNN layers
        input_layer = Rectangle(width=1, height=1, color=BLUE)
        conv_layer_1 = Rectangle(width=1.5, height=1.5, color=GREEN)
        pool_layer_1 = Rectangle(width=1, height=1, color=ORANGE)
        conv_layer_2 = Rectangle(width=1, height=1, color=GREEN)
        pool_layer_2 = Rectangle(width=0.5, height=0.5, color=ORANGE)
        dense_layer = Rectangle(width=0.5, height=0.5, color=PURPLE)
        output_layer = Rectangle(width=0.5, height=0.5, color=RED)

        # Label the final output
        output_label = Text("Final Categorization").next_to(output_layer, RIGHT)

        # Arrange layers
        layers = VGroup(input_layer, conv_layer_1, pool_layer_1, conv_layer_2, pool_layer_2, dense_layer, output_layer)
        layers.arrange(RIGHT, buff=0.2)

        # Add layers to scene
        self.play(*[Create(layer) for layer in layers])
        self.play(Write(output_label))