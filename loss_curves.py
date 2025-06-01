from manim import *
import pandas as pd
import numpy as np

class LossCurves(Scene):
    def construct(self):
        # === Title ===
        title = Text("Training and Validation Loss Curves", font_size=48, color=BLUE)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title, run_time=2))

        # === Load CSV data ===
        df = pd.read_csv("results.csv")
        epochs = list(range(len(df)))

        # Extract loss values
        train_box = df["train/box_loss"].values
        val_box = df["val/box_loss"].values
        train_cls = df["train/cls_loss"].values
        val_cls = df["val/cls_loss"].values
        train_dfl = df["train/dfl_loss"].values
        val_dfl = df["val/dfl_loss"].values

        # Common x values (downsampled to reduce clutter)
        x_vals = epochs[::4]
        def downsample(y): return y[::4]

        # Function to show each loss pair
        def show_loss_pair(train_y, val_y, train_label_text, val_label_text, train_color, val_color, narration_text):
            # Dynamic max Y for scaling
            y_max = max(max(train_y), max(val_y))
            axes = Axes(
                x_range=[0, len(df), 10],
                y_range=[0, y_max + 0.2, 0.2],
                x_length=10,
                y_length=5,
                axis_config={"include_tip": False}
            ).center().shift(DOWN * 0.5)
            axes.add_coordinates(font_size=20)

            self.play(Create(axes, run_time=2))

            train_curve = axes.plot_line_graph(
                x_values=x_vals,
                y_values=downsample(train_y),
                line_color=train_color,
                vertex_dot_radius=0.01
            )
            val_curve = axes.plot_line_graph(
                x_values=x_vals,
                y_values=downsample(val_y),
                line_color=val_color,
                vertex_dot_radius=0.01
            )

            train_label = Text(train_label_text, font_size=24, color=train_color)
            val_label = Text(val_label_text, font_size=24, color=val_color)

            train_label.next_to(train_curve, UP, buff=0.3)
            val_label.next_to(val_curve, RIGHT, buff=0.3)

            narration = Paragraph(
                narration_text,
                font_size=24,
                width=10
            ).to_edge(DOWN, buff=0.6)

            self.play(Create(train_curve, run_time=3), FadeIn(train_label, run_time=3))
            self.play(Create(val_curve, run_time=3), FadeIn(val_label, run_time=3), Write(narration, run_time=3))
            self.wait(4)
            self.play(*[FadeOut(m) for m in [axes, train_curve, val_curve, train_label, val_label, narration]], run_time=1.5)

        # === Narration Intro ===
        narration1 = Paragraph(
            "Let's start by visualizing how the loss evolves during training.",
            font_size=24, width=10
        ).to_edge(DOWN, buff=0.6)
        self.play(Write(narration1, run_time=3))
        self.wait(3)
        self.play(FadeOut(narration1, run_time=1.5))

        # === Plot Box Loss ===
        show_loss_pair(
            train_y=train_box,
            val_y=val_box,
            train_label_text="Train Box Loss",
            val_label_text="Val Box Loss",
            train_color=GREEN,
            val_color=RED,
            narration_text="Green shows training box loss, red shows validation box loss."
        )

        # === Plot Classification Loss ===
        show_loss_pair(
            train_y=train_cls,
            val_y=val_cls,
            train_label_text="Train Cls Loss",
            val_label_text="Val Cls Loss",
            train_color=BLUE,
            val_color=ORANGE,
            narration_text="Now we see the classification loss. Training is in blue, validation in orange."
        )

        # === Plot DFL Loss ===
        show_loss_pair(
            train_y=train_dfl,
            val_y=val_dfl,
            train_label_text="Train DFL Loss",
            val_label_text="Val DFL Loss",
            train_color=YELLOW,
            val_color=PURPLE,
            narration_text="Lastly, the DFL loss. Yellow for training, purple for validation."
        )

        # === Outro ===
        outro = Paragraph(
            "These loss curves help us understand model performance and generalization.",
            font_size=24,
            width=10
        ).to_edge(DOWN, buff=0.6)
        self.play(Write(outro, run_time=3))
        self.wait(4)
        self.play(FadeOut(outro, run_time=1.5), FadeOut(title, run_time=1.5))
