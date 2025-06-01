from manim import *
import pandas as pd
import numpy as np

class LossCurves(Scene):
    def construct(self):
        # === Title ===
        title = Text("Training and Validation Loss Curves", font_size=48, color=BLUE)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title, run_time=2))

        # === Load Data ===
        df = pd.read_csv("results.csv")
        epochs = list(range(len(df)))
        x_vals = np.array(epochs)

        # Extract loss values
        train_box = df["train/box_loss"].values
        val_box = df["val/box_loss"].values
        train_cls = df["train/cls_loss"].values
        val_cls = df["val/cls_loss"].values
        train_dfl = df["train/dfl_loss"].values
        val_dfl = df["val/dfl_loss"].values

        # Helper: plot with animation
        def make_animated_curve(axes, x_vals, y_vals, color):
            curve = axes.plot(
                lambda x: np.interp(x, x_vals, y_vals),
                x_range=[x_vals[0], x_vals[-1]],
                color=color,
                use_smoothing=False
            )
            anim = ShowCreation(curve, run_time=4)
            return curve, anim

        # Helper: render each curve pair
        def show_loss_pair(train_y, val_y, train_label_text, val_label_text, train_color, val_color, narration_text):
            y_max = max(np.max(train_y), np.max(val_y))
            y_step = 0.5 if y_max > 2 else 0.2

            axes = Axes(
                x_range=[0, len(df), 10],
                y_range=[0, y_max + 0.3, y_step],
                x_length=10,
                y_length=5,
                axis_config={"include_tip": False}
            ).center().shift(DOWN * 0.25)
            axes.add_coordinates(font_size=20)

            self.play(Create(axes, run_time=2))

            train_curve, anim_train = make_animated_curve(axes, x_vals, train_y, train_color)
            val_curve, anim_val = make_animated_curve(axes, x_vals, val_y, val_color)

            train_label = Text(train_label_text, font_size=24, color=train_color).next_to(axes, UP, buff=0.3)
            val_label = Text(val_label_text, font_size=24, color=val_color).next_to(axes, RIGHT, buff=0.3)

            narration = Paragraph(
                narration_text,
                font_size=22, width=10
            ).next_to(axes, DOWN, buff=1.0)

            self.play(anim_train, FadeIn(train_label, run_time=2))
            self.play(anim_val, FadeIn(val_label, run_time=2), Write(narration, run_time=3))
            self.wait(3)

            self.play(
                *[FadeOut(m) for m in [axes, train_curve, val_curve, train_label, val_label, narration]],
                run_time=1.5
            )

        # === Intro ===
        narration1 = Paragraph(
            "Let's start by visualizing how the loss evolves during training.",
            font_size=22, width=10
        ).next_to(ORIGIN, DOWN * 3.5)
        self.play(Write(narration1, run_time=3))
        self.wait(3)
        self.play(FadeOut(narration1, run_time=1.5))

        # === Show box loss ===
        show_loss_pair(
            train_y=train_box,
            val_y=val_box,
            train_label_text="Train Box Loss",
            val_label_text="Val Box Loss",
            train_color=GREEN,
            val_color=RED,
            narration_text="Green shows training box loss, red shows validation box loss."
        )

        # === Show classification loss ===
        show_loss_pair(
            train_y=train_cls,
            val_y=val_cls,
            train_label_text="Train Cls Loss",
            val_label_text="Val Cls Loss",
            train_color=BLUE,
            val_color=ORANGE,
            narration_text="Now we see the classification loss. Training is in blue, validation in orange."
        )

        # === Show DFL loss ===
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
            font_size=22, width=10
        ).next_to(ORIGIN, DOWN * 3.5)
        self.play(Write(outro, run_time=3))
        self.wait(3)
        self.play(FadeOut(outro, run_time=1.5), FadeOut(title, run_time=1.5))
