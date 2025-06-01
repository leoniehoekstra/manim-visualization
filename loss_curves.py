from manim import *
import pandas as pd

class LossCurves(Scene):
    def construct(self):
        # === Title ===
        title = Text("Training and Validation Loss Curves", font_size=48, color=BLUE)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))

        # === Load CSV data ===
        df = pd.read_csv("results.csv")
        epochs = range(len(df))

        # Extract loss values
        train_box = df["train/box_loss"].values
        val_box = df["val/box_loss"].values
        train_cls = df["train/cls_loss"].values
        val_cls = df["val/cls_loss"].values
        train_dfl = df["train/dfl_loss"].values
        val_dfl = df["val/dfl_loss"].values

        # === Narration 1 ===
        narration1 = Paragraph(
            "Let's start by visualizing how the loss evolves during training.",
            font_size=24, width=10
        ).to_edge(DOWN, buff=0.5)
        self.play(Write(narration1))
        self.wait(2)
        self.play(FadeOut(narration1))
        

        # === Axes ===
        max_loss = max(
            max(train_box), max(val_box),
            max(train_cls), max(val_cls),
            max(train_dfl), max(val_dfl)
        )
        
        axes = Axes(
            x_range=[0, len(df), 5],
            y_range=[0, max_loss + 0.5, 0.5],
            x_length=10,
            y_length=5,
            axis_config={"include_tip": False}
        ).center().shift(DOWN * 0.5)
        
        axes.add_coordinates()
        

        x_vals = list(epochs)

        self.play(Create(axes))

        # === Plot Box Loss ===
        train_box_curve = axes.plot_line_graph(
            x_values=x_vals,
            y_values=train_box,
            line_color=GREEN
        )
        val_box_curve = axes.plot_line_graph(
            x_values=x_vals,
            y_values=val_box,
            line_color=RED
        )

        train_box_label = Text("Train Box Loss", font_size=24, color=GREEN).next_to(train_box_curve, UP, buff=0.3)
        val_box_label = Text("Val Box Loss", font_size=24, color=RED).next_to(val_box_curve, RIGHT, buff=0.3)

        narration2 = Paragraph(
            "Green shows training box loss, red shows validation box loss.",
            font_size=24, width=10
        ).to_edge(DOWN, buff=0.5)

        self.play(Create(train_box_curve), FadeIn(train_box_label))
        self.play(Create(val_box_curve), FadeIn(val_box_label), Write(narration2))
        self.wait(3)
        self.play(FadeOut(train_box_curve), FadeOut(val_box_curve), FadeOut(train_box_label), FadeOut(val_box_label), FadeOut(narration2))

        # === Plot Class Loss ===
        train_cls_curve = axes.plot_line_graph(
            x_values=x_vals,
            y_values=train_cls,
            line_color=BLUE
        )
        val_cls_curve = axes.plot_line_graph(
            x_values=x_vals,
            y_values=val_cls,
            line_color=ORANGE
        )

        train_cls_label = Text("Train Cls Loss", font_size=24, color=BLUE).next_to(train_cls_curve, UP, buff=0.3)
        val_cls_label = Text("Val Cls Loss", font_size=24, color=ORANGE).next_to(val_cls_curve, RIGHT, buff=0.3)

        narration3 = Paragraph(
            "Now we see the classification loss. Training is in blue, validation in orange.",
            font_size=24, width=10
        ).to_edge(DOWN, buff=0.5)

        self.play(Create(train_cls_curve), FadeIn(train_cls_label))
        self.play(Create(val_cls_curve), FadeIn(val_cls_label), Write(narration3))
        self.wait(3)
        self.play(FadeOut(train_cls_curve), FadeOut(val_cls_curve), FadeOut(train_cls_label), FadeOut(val_cls_label), FadeOut(narration3))

        # === Plot DFL Loss ===
        train_dfl_curve = axes.plot_line_graph(
            x_values=x_vals,
            y_values=train_dfl,
            line_color=YELLOW
        )
        val_dfl_curve = axes.plot_line_graph(
            x_values=x_vals,
            y_values=val_dfl,
            line_color=PURPLE
        )

        train_dfl_label = Text("Train DFL Loss", font_size=24, color=YELLOW).next_to(train_dfl_curve, UP, buff=0.3)
        val_dfl_label = Text("Val DFL Loss", font_size=24, color=PURPLE).next_to(val_dfl_curve, RIGHT, buff=0.3)

        narration4 = Paragraph(
            "Lastly, the DFL loss — yellow for training, purple for validation.",
            font_size=24, width=10
        ).to_edge(DOWN, buff=0.5)

        self.play(Create(train_dfl_curve), FadeIn(train_dfl_label))
        self.play(Create(val_dfl_curve), FadeIn(val_dfl_label), Write(narration4))
        self.wait(3)
        self.play(FadeOut(train_dfl_curve), FadeOut(val_dfl_curve), FadeOut(train_dfl_label), FadeOut(val_dfl_label), FadeOut(narration4))

        # Outro
        outro = Paragraph(
            "These loss curves help us understand model performance and generalization.",
            font_size=24, width=10
        ).to_edge(DOWN, buff=0.5)
        self.play(Write(outro))
        self.wait(3)
        self.play(FadeOut(outro), FadeOut(axes), FadeOut(title))
