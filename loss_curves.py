from manim import *
import pandas as pd
import numpy as np

class LossCurves(Scene):
    def construct(self):
        # === Title ===
        title = Text("Training and Validation Loss Curves", font_size=48, color=BLUE)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))

        # === Load CSV data ===
        df = pd.read_csv("results.csv")
        epochs = range(len(df))
        x_vals = list(epochs)

        # Extract loss values
        train_box = df["train/box_loss"].values
        val_box = df["val/box_loss"].values
        train_cls = df["train/cls_loss"].values
        val_cls = df["val/cls_loss"].values
        train_dfl = df["train/dfl_loss"].values
        val_dfl = df["val/dfl_loss"].values

        # Helper: create smooth plot using interpolation
        def make_curve(axes, x_vals, y_vals, color):
            return axes.plot(
                lambda x: np.interp(x, x_vals, y_vals),
                x_range=[x_vals[0], x_vals[-1]],
                color=color
            )

        # === Function to render each loss pair ===
        def show_loss_pair(train_vals, val_vals, train_color, val_color, train_label_text, val_label_text, narration_text):
            y_max = max(max(train_vals), max(val_vals)) + 0.2
            axes = Axes(
                x_range=[0, len(df), 10],
                y_range=[0, y_max, round(y_max / 10, 2)],
                x_length=10,
                y_length=5,
                axis_config={"include_tip": False}
            ).center().shift(DOWN * 0.5)
            axes.add_coordinates(font_size=20)

            self.play(Create(axes))

            train_curve = make_curve(axes, x_vals, train_vals, train_color)
            val_curve = make_curve(axes, x_vals, val_vals, val_color)

            train_label = Text(train_label_text, font_size=24, color=train_color).next_to(train_curve, UP, buff=0.3)
            val_label = Text(val_label_text, font_size=24, color=val_color).next_to(val_curve, RIGHT, buff=0.3)

            narration = Paragraph(
                narration_text,
                font_size=22,
                width=10
            ).next_to(axes, DOWN, buff=0.6)

            self.play(Create(train_curve), FadeIn(train_label))
            self.play(Create(val_curve), FadeIn(val_label), Write(narration))
            self.wait(3)

            self.play(
                FadeOut(train_curve), FadeOut(val_curve),
                FadeOut(train_label), FadeOut(val_label),
                FadeOut(narration), FadeOut(axes)
            )

        # === Narration 1 ===
        intro = Paragraph(
            "Let's start by visualizing how the loss evolves during training.",
            font_size=22, width=10
        ).move_to(DOWN * 3)
        self.play(Write(intro))
        self.wait(2)
        self.play(FadeOut(intro))

        # === Show box loss ===
        show_loss_pair(
            train_vals=train_box,
            val_vals=val_box,
            train_color=GREEN,
            val_color=RED,
            train_label_text="Train Box Loss",
            val_label_text="Val Box Loss",
            narration_text="Green shows training box loss, red shows validation box loss."
        )

        # === Show classification loss ===
        show_loss_pair(
            train_vals=train_cls,
            val_vals=val_cls,
            train_color=BLUE,
            val_color=ORANGE,
            train_label_text="Train Cls Loss",
            val_label_text="Val Cls Loss",
            narration_text="Now we see the classification loss. Training is in blue, validation in orange."
        )

        # === Show DFL loss ===
        show_loss_pair(
            train_vals=train_dfl,
            val_vals=val_dfl,
            train_color=YELLOW,
            val_color=PURPLE,
            train_label_text="Train DFL Loss",
            val_label_text="Val DFL Loss",
            narration_text="Lastly, the DFL loss — yellow for training, purple for validation."
        )

        # === Outro ===
        outro = Paragraph(
            "These loss curves help us understand model performance and generalization.",
            font_size=22,
            width=10
        ).move_to(DOWN * 3)
        self.play(Write(outro))
        self.wait(3)
        self.play(FadeOut(outro), FadeOut(title))
