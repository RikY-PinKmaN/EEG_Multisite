#!/usr/bin/env python3
"""Render the diagram slides as PNGs and build a downloadable 16:9 PowerPoint deck."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
OUT = os.path.join(HERE, "TIMING_PRESENTATION.pptx")
SHE, LDS, MAN = "#d1495b", "#3b7dd8", "#37a66b"
ACC, INK, GREY = "#0f6d7a", "#131a24", "#5f6c7a"
plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})


def _box(ax, x, y, w, h, label, sub="", ec=INK, lw=1.6, fs=12, fc="none"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                ec=ec, fc=fc, lw=lw))
    ax.text(x + w / 2, y + h * (.62 if sub else .5), label, ha="center", va="center",
            fontsize=fs, weight="bold", color=ec if ec != INK else INK)
    if sub:
        ax.text(x + w / 2, y + h * .28, sub, ha="center", va="center",
                fontsize=fs - 2.4, color=GREY)


def _arrow(ax, x1, y1, x2, y2, color=GREY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
                                 lw=1.5, color=color, shrinkA=0, shrinkB=0))


def _blank(w=13.0, h=5.2):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    return fig, ax


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)
    return p


# ---- D1: study design ----
fig, ax = _blank(13.0, 3.4)
ax.plot([6, 94], [55, 55], color="#d0d8e0", lw=2.5, zorder=0)
for x, lab, col, note in [(12, "SHE", SHE, "visit 1"), (38, "LDS", LDS, "visit 2"),
                          (64, "SHE", SHE, "repeat"), (90, "MAN", MAN, "visit 4")]:
    ax.scatter([x], [55], s=2100, color=col, zorder=2)
    ax.text(x, 55, lab, ha="center", va="center", color="white", fontsize=13, weight="bold")
    ax.text(x, 28, note, ha="center", fontsize=11.5, color=GREY)
ax.text(3, 86, "The same 2 subjects travel to all four sessions", fontsize=14, color=INK)
ax.text(3, 8, "Repeating Sheffield separates “this laboratory” from “this day”. "
              "Different amplifier, cap and reference at each site.",
        fontsize=11.5, color=GREY)
save(fig, "d_design.png")

# ---- D2: photodiode subtraction ----
fig, ax = _blank(13.0, 3.6)
ax.plot([2, 98], [72, 72], color="#d0d8e0", lw=2.2)
ax.plot([2, 98], [30, 30], color="#d0d8e0", lw=2.2)
ax.text(2, 88, "What the computer wrote down", fontsize=13.5, color=INK)
ax.plot([26, 26], [63, 81], color=SHE, lw=3.4); ax.scatter([26], [72], s=90, color=SHE, zorder=3)
ax.text(2, 12, "When the screen actually changed  (light sensor)", fontsize=13.5, color=INK)
ax.plot([50, 50], [21, 39], color=MAN, lw=3.4); ax.scatter([50], [30], s=90, color=MAN, zorder=3)
ax.plot([26, 26], [30, 72], color="#98a4b0", ls=":", lw=1.4)
ax.annotate("", xy=(50, 51), xytext=(26, 51),
            arrowprops=dict(arrowstyle="<->", color=ACC, lw=2.2))
ax.text(38, 56, "the error", ha="center", fontsize=14, color=ACC, weight="bold")
ax.text(58, 55, "error  =  true onset  −  logged trigger", fontsize=14, color=INK,
        family="monospace")
ax.text(58, 43, "measured on every single trial", fontsize=12, color=GREY)
save(fig, "d_measure.png")

# ---- D3: preprocessing ----
fig, ax = _blank(13.0, 4.0)
steps = [("Same 14 sensors", "common to all sites"), ("Common reference", "average of both earlobes"),
         ("Filter 0.5–30 Hz", "drift and mains removed"), ("Resample 256 Hz", "one common clock")]
for i, (a, b) in enumerate(steps):
    x = 1 + i * 25
    _box(ax, x, 68, 22, 24, a, b)
    if i < 3:
        _arrow(ax, x + 22.6, 80, x + 24.6, 80)
ax.plot([87, 87, 46], [68, 58, 58], color="#98a4b0", lw=1.5)
_arrow(ax, 46, 58, 44, 58)
_box(ax, 20, 24, 24, 24, "Cut at the", "TRUE onset", ec=MAN, lw=2)
_box(ax, 56, 24, 24, 24, "Cut at the", "LOGGED trigger", ec=SHE, lw=2)
ax.text(50, 13, "Identical recording, identical trials — only the cutting point differs",
        ha="center", fontsize=12.5, color=INK)
ax.text(50, 4, "Window −200 to +950 ms · baseline corrected · kept only if clean under BOTH cuts",
        ha="center", fontsize=11.5, color=GREY)
save(fig, "d_prep.png")


# ============================ the deck ============================
SLIDES = [
    dict(kind="cover",
         title="How wrong is the clock in an EEG experiment?",
         sub="Every EEG study assumes the recorded timestamp marks the moment the stimulus "
             "appeared on screen. We put a light sensor on the screen at three laboratories "
             "and measured the gap — on all 6,718 trials.",
         foot="Sourojit Goswami  ·  2 subjects · 4 sessions · 3 sites · 6,718 trials"),
    dict(tag="01 — THE DATA",
         title="The same people, the same battery, three different laboratories",
         sub="Seven recordings were made per visit. Two of them can carry a timing measurement.",
         img="d_design.png",
         foot="USED: n170 (faces vs cars) and p3b (rare targets) — both have two balanced picture "
              "conditions and a sharp landmark in time.   NOT USED: n2pc (comparison is left vs "
              "right, no time landmark) · resting (no stimuli) · blinks and eog (recorded to "
              "characterise eye artifacts) · assr (sound — a light sensor cannot see sound)."),
    dict(tag="02 — THE MEASUREMENT",
         title="The error is subtracted, not estimated",
         sub="A photodiode taped to the screen, recorded on the same amplifier as the brain "
             "signal, fires the instant the screen actually changes.",
         img="d_measure.png",
         foot="Leeds 23.9 ms · Manchester 78.5 ms · Sheffield 91.6 ms.   Both timestamps sit on "
              "the same recording clock, so the error is a direct subtraction on every trial — "
              "no model, no assumption."),
    dict(tag="03 — PREPROCESSING",
         title="Every step applied identically to both versions of the data",
         sub="Three sites, three different amplifiers and reference points — first brought onto "
             "common ground, then each trial cut twice from the same signal.",
         img="d_prep.png",
         foot="The control that makes the comparison fair: a trial is kept only if it passes "
              "quality checks under BOTH cuts. Otherwise the two versions would be built from "
              "different trials, and any difference could come from trial selection rather than "
              "timing."),
    dict(tag="04 — THE RAW MEASUREMENT",
         title="The error is large, steady within a session, and different at every site",
         sub="One dot per trial. Height is how late the timestamp was.",
         img="p1_raw_trials.png",
         foot="Within any one block the spread is only a few milliseconds — this is a steady "
              "offset, not random noise. Averaging trials removes noise; it does nothing to an "
              "offset."),
    dict(tag="05 — FINDING ONE",
         title="The error belongs to the laboratory, not to the trial",
         sub="Splitting the total variability by level: to predict a trial's timing error, what "
             "do you most need to know?",
         img="p2_variance.png",
         foot="Site accounts for 77% of the variance; the trial level — the only part averaging "
              "could remove — accounts for about 1%."),
    dict(tag="06 — FINDING TWO",
         title="The brain response moves by exactly that amount",
         sub="Re-cutting the identical recording at the uncorrected timestamp and re-averaging.",
         img="p3_erp_shift.png",
         foot="The EEG and the light sensor agree to 1.21 ms on average across 16 comparisons — "
              "an independent confirmation of the whole processing chain. Waveform shape survives: "
              "correlation above 0.97 everywhere."),
    dict(tag="07 — THE ACTUAL BRAIN RESPONSES",
         title="The response slides out of the measurement window",
         sub="Solid = cut at the true onset. Dashed = cut at the logged trigger. Shaded band is "
             "the time window a laboratory measures in.",
         img="p5_erp_waveforms.png",
         foot="The dashed traces are the same waveform arriving late. At Sheffield the P3b is "
              "displaced far enough that much of it has left the measurement window entirely — "
              "which is how a real, healthy brain response gets recorded as a weak or absent one."),
    dict(tag="08 — WHAT IT COSTS",
         title="Bad timing costs real decoding accuracy",
         sub="AUC — the area under the ROC curve — scores how well the two conditions can be told "
             "apart. 0.5 is a coin flip, 1.0 is perfect.",
         img="p6_auc.png",
         foot="For N170 with all trials, correct timing reaches AUC 0.87 while the uncorrected "
              "version sits at 0.48 — below chance, and getting worse as more training data is "
              "added. Deliberately teaching the classifier that timing varies recovers about half "
              "the loss (0.69), but does not restore it."),
    dict(kind="close",
         title="What this shows",
         bullets=[
             "Stimulus timing error at these three laboratories runs from 24 to 135 ms.",
             "It is a per-session constant, not trial-to-trial jitter: site 77% of the variance, "
             "trial about 1%.",
             "It displaces the brain response by exactly that amount while leaving its shape "
             "intact — confirmed against the light sensor to about 1 ms.",
             "It changes the number a laboratory reports, flipping its sign in 4 of 16 "
             "comparisons, and costs up to 0.39 AUC in decoding.",
             "Limits: two subjects, three specific laboratories. These are not a lookup table "
             "for other labs — they are evidence that the error is large and site-specific.",
         ],
         foot="Recommendation: measure your own timing. A photodiode on a spare amplifier channel "
              "turns an unknown into a measured quantity."),
]

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
SW, SH = 13.333, 7.5
blank = prs.slide_layouts[6]


def tb(slide, x, y, w, h, text, size, bold=False, color=INK, align=PP_ALIGN.LEFT,
       space=1.16, italic=False):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    para = tf.paragraphs[0]
    para.alignment = align
    para.line_spacing = space
    run = para.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = "Calibri"
    run.font.color.rgb = RGBColor.from_string(color.lstrip("#").upper())
    return box


for i, s in enumerate(SLIDES):
    sl = prs.slides.add_slide(blank)
    if s.get("kind") == "cover":
        tb(sl, .95, 1.55, 11.4, .4, "TRAVELING-SUBJECTS MULTI-SITE EEG", 12, True, ACC)
        tb(sl, .95, 2.15, 11.0, 1.6, s["title"], 40, True, INK)
        tb(sl, .95, 3.95, 10.2, 1.6, s["sub"], 16, False, GREY, space=1.3)
        tb(sl, .95, 6.25, 11.4, .5, s["foot"], 13, False, GREY)
        continue
    if s.get("kind") == "close":
        tb(sl, .95, .78, 11.4, .8, s["title"], 30, True, INK)
        y = 1.85
        for b in s["bullets"]:
            tb(sl, 1.15, y, 11.0, .8, "•  " + b, 15, False, INK, space=1.25)
            y += .78
        tb(sl, .95, 6.45, 11.4, .7, s["foot"], 13, False, ACC, space=1.2)
        continue

    tb(sl, .72, .52, 11.9, .35, s["tag"], 11.5, True, ACC)
    tb(sl, .72, .95, 11.9, 1.05, s["title"], 25, True, INK)
    tb(sl, .72, 2.06, 11.6, .66, s["sub"], 13, False, GREY, space=1.22)

    top, avail_h = 2.86, 3.16
    im = os.path.join(FIG, s["img"])
    from PIL import Image
    iw, ih = Image.open(im).size
    ar = iw / ih
    w = min(12.3, avail_h * ar)
    h = w / ar
    sl.shapes.add_picture(im, Inches((SW - w) / 2), Inches(top + (avail_h - h) / 2),
                          width=Inches(w))

    line = sl.shapes.add_textbox(Inches(.72), Inches(6.12), Inches(11.9), Inches(.02))
    tb(sl, .72, 6.20, 11.9, 1.0, s["foot"], 12, False, GREY, space=1.22)

prs.save(OUT)
print("\nwrote", OUT, "%.1f KB · %d slides" % (os.path.getsize(OUT) / 1024, len(SLIDES)))
