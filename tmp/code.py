from manim import *
import numpy as np
from manim_ml.neural_network import NeuralNetwork
from manim_ml.neural_network.layers import Convolutional2DLayer, FeedForwardLayer
from manim_ml.neural_network.layers.parent_layers import NeuralNetworkLayer

class CNNBasicStructure(Scene):
    def construct(self):
        input_layer = Rectangle(width=1, height=3, color=BLUE)
        input_label = Text("Input Layer").scale(0.5).next_to(input_layer, DOWN)
        hidden_layer1 = Rectangle(width=1, height=3, color=GREEN).next_to(input_layer, RIGHT, buff=1)
        hidden_label1 = Text("Hidden Layer 1").scale(0.5).next_to(hidden_layer1, DOWN)
        hidden_layer2 = Rectangle(width=1, height=3, color=GREEN).next_to(hidden_layer1, RIGHT, buff=1)
        hidden_label2 = Text("Hidden Layer 2").scale(0.5).next_to(hidden_layer2, DOWN)
        output_layer = Rectangle(width=1, height=3, color=RED).next_to(hidden_layer2, RIGHT, buff=1)
        output_label = Text("Output Layer").scale(0.5).next_to(output_layer, DOWN)
        self.play(Create(input_layer), Write(input_label))
        self.play(Create(hidden_layer1), Write(hidden_label1))
        self.play(Create(hidden_layer2), Write(hidden_label2))
        self.play(Create(output_layer), Write(output_label))
        input_to_hidden1 = Arrow(input_layer.get_right(), hidden_layer1.get_left(), buff=0.1)
        hidden1_to_hidden2 = Arrow(hidden_layer1.get_right(), hidden_layer2.get_left(), buff=0.1)
        hidden2_to_output = Arrow(hidden_layer2.get_right(), output_layer.get_left(), buff=0.1)
        self.play(Create(input_to_hidden1))
        self.play(Create(hidden1_to_hidden2))
        self.play(Create(hidden2_to_output))

class ConvLayerVisualization(Scene):
    def construct(self):
        input_matrix = np.array([
            [0.2, 0.4, 0.5, 0.6, 0.3, 0.2],
            [0.1, 0.9, 0.8, 0.7, 0.2, 0.1],
            [0.3, 0.7, 0.6, 0.5, 0.4, 0.3],
            [0.4, 0.6, 0.5, 0.8, 0.7, 0.3],
            [0.5, 0.4, 0.3, 0.2, 0.1, 0.2],
            [0.6, 0.7, 0.8, 0.9, 0.5, 0.4]
        ])
        input_image = Matrix(input_matrix, v_buff=0.5)
        input_image.scale(0.5)
        input_label = Text("Input Image", font_size=24).next_to(input_image, UP, buff=0.1)
        filter_matrix = np.array([
            [1, 0, -1],
            [0, 0, 0],
            [-1, 0, 1]
        ])
        conv_filter = Matrix(filter_matrix, v_buff=0.5).scale(0.5).next_to(input_image, RIGHT, buff=0.5)
        filter_label = Text("Filter", font_size=24).next_to(conv_filter, UP, buff=0.1)
        output_matrix = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ])
        feature_map = Matrix(output_matrix, v_buff=0.5).scale(0.5).next_to(conv_filter, RIGHT, buff=0.5)
        feature_map_label = Text("Feature Map", font_size=24).next_to(feature_map, UP, buff=0.1)
        self.play(Create(input_image), Write(input_label))
        self.play(Create(conv_filter), Write(filter_label))
        self.play(Create(feature_map), Write(feature_map_label))
        anims = []
        for i in range(4):
            for j in range(4):
                rect = Rectangle(width=1.5, height=1.5)
                rect.move_to(input_image.get_columns()[j][i].get_center())
                anims.append(ApplyMethod(rect.set_color, BLUE))
                anims.append(ApplyMethod(rect.set_color, WHITE))
        self.play(*anims, run_time=4)
        self.wait(2)

class PoolingLayerScene(Scene):
    def construct(self):
        feature_map = VGroup(*[Square(side_length=0.5) for _ in range(25)]).arrange_in_grid(rows=5, cols=5, buff=0.1)
        original_label = Text("Original Feature Map").next_to(feature_map, UP)
        pooled_map = VGroup(*[Square(side_length=0.5, color=RED) for _ in range(4)]).arrange_in_grid(rows=2, cols=2, buff=0.1)
        pooled_label = Text("Pooled Feature Map").next_to(pooled_map, UP)
        pooled_map.next_to(feature_map, RIGHT, buff=1)
        self.play(Create(feature_map), Write(original_label))
        self.wait(1)
        self.play(
            feature_map.animate.shift(LEFT * 2),
            Transform(feature_map, pooled_map),
            Write(pooled_label)
        )
        self.wait(2)

class FullyConnectedLayerScene(Scene):
    def construct(self):
        neural_network = NeuralNetwork([
            Convolutional2DLayer(20, 3, 3),
            FeedForwardLayer(10),
        ])
        self.play(Create(neural_network))
        title = Text("Fully Connected Layer and Output").to_edge(UP)
        self.play(Write(title))
        self.wait(2)

class CNNOverview(Scene):
    def construct(self):
        input_layer = NeuralNetworkLayer(input_shape=(28, 28), layer_type="Input")
        conv_layer = NeuralNetworkLayer(input_shape=(28, 28), filter_count=32, layer_type="Convolution")
        pool_layer = NeuralNetworkLayer(input_shape=(14, 14), layer_type="Pooling")
        fc_layer = NeuralNetworkLayer(input_units=128, layer_type="FullyConnected")
        output_layer = NeuralNetworkLayer(output_units=10, layer_type="Output")
        cnn = NeuralNetwork([
            input_layer, conv_layer, pool_layer, fc_layer, output_layer
        ], layer_spacing=0.5)
        self.add(cnn)
        self.play(FadeIn(cnn))
        title = Text("Convolutional Neural Network Architecture", font_size=48).to_edge(UP)
        self.add(title)
        self.wait(2)