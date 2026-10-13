#!/usr/bin/env python3
"""Build the plain-language explainer PDF for the marker-timing work (no GAN content)."""
import os
import pandas as pd
from fpdf import FPDF

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
DJV = "/usr/share/fonts/truetype/dejavu"
OUT = os.path.join(HERE, "TIMING_EXPLAINER.pdf")

INK = (25, 30, 40)
GREY = (95, 108, 122)
ACCENT = (14, 95, 108)
RULE = (208, 216, 224)


class Doc(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("DJV", "", 7.5)
        self.set_text_color(*GREY)
        self.set_y(9)
        self.cell(0, 4, "Marker timing error in multi-site EEG", align="L")
        self.set_y(9)
        self.cell(0, 4, "page %d" % self.page_no(), align="R")
        self.set_draw_color(*RULE)
        self.line(18, 15, 192, 15)
        self.set_y(22)

    def footer(self):
        pass


d = Doc(orientation="P", unit="mm", format="A4")
d.set_auto_page_break(True, margin=18)
d.add_font("DJV", "", f"{DJV}/DejaVuSans.ttf")
d.add_font("DJV", "B", f"{DJV}/DejaVuSans-Bold.ttf")
d.add_font("DJV", "I", f"{DJV}/DejaVuSans-Oblique.ttf")
d.add_font("DJVM", "", f"{DJV}/DejaVuSansMono.ttf")
d.set_margins(18, 18, 18)
W = 174


def h1(t):
    d.ln(3)
    d.set_font("DJV", "B", 15.5)
    d.set_text_color(*INK)
    d.multi_cell(W, 7.2, t)
    d.set_draw_color(*ACCENT)
    d.set_line_width(0.7)
    y = d.get_y() + 1.4
    d.line(18, y, 18 + 26, y)
    d.set_line_width(0.2)
    d.ln(4.5)


def h2(t):
    d.ln(2.2)
    d.set_font("DJV", "B", 11.3)
    d.set_text_color(*INK)
    d.multi_cell(W, 5.6, t)
    d.ln(1.4)


def p(t, size=9.9, gap=4.9):
    d.set_font("DJV", "", size)
    d.set_text_color(*INK)
    d.multi_cell(W, gap, t)
    d.ln(2.1)


def term(name, body):
    d.set_font("DJV", "B", 9.9)
    d.set_text_color(*ACCENT)
    d.multi_cell(W, 4.9, name)
    d.set_font("DJV", "", 9.9)
    d.set_text_color(*INK)
    d.multi_cell(W, 4.9, body)
    d.ln(2.4)


def bullet(t, size=9.9):
    d.set_font("DJV", "", size)
    d.set_text_color(*INK)
    x = d.get_x()
    d.cell(4.5, 4.8, "•")
    d.multi_cell(W - 4.5, 4.8, t)
    d.set_x(x)
    d.ln(1.0)


def note(t):
    y0 = d.get_y()
    d.set_font("DJV", "", 9.3)
    d.set_text_color(*GREY)
    d.set_x(23)
    d.multi_cell(W - 5, 4.7, t)
    d.set_draw_color(*ACCENT)
    d.set_line_width(0.8)
    d.line(19.2, y0 + 0.6, 19.2, d.get_y() - 0.6)
    d.set_line_width(0.2)
    d.set_x(18)
    d.ln(3.2)


def figure(name, caption, width=W):
    path = os.path.join(FIG, name)
    if d.get_y() > 205:
        d.add_page()
    d.image(path, x=18, w=width)
    d.ln(1.6)
    d.set_font("DJV", "I", 8.6)
    d.set_text_color(*GREY)
    d.multi_cell(W, 4.3, caption)
    d.ln(3.4)


# ============================ COVER ============================
d.add_page()
d.ln(26)
d.set_font("DJV", "", 8.6)
d.set_text_color(*ACCENT)
d.cell(0, 5, "TRAVELING-SUBJECTS MULTI-SITE EEG  ·  METHODS EXPLAINER")
d.ln(13)
d.set_font("DJV", "B", 23)
d.set_text_color(*INK)
d.multi_cell(W, 10.4, "How wrong is the clock in an EEG experiment?")
d.ln(5)
d.set_font("DJV", "", 11.4)
d.set_text_color(*GREY)
d.multi_cell(W, 6.0,
    "Every EEG study assumes the timestamp it records marks the moment the stimulus appeared "
    "on the screen. We put a light sensor on the screen at three laboratories and measured how "
    "far off that assumption is — on all 6,718 trials.")
d.ln(9)
d.set_draw_color(*RULE)
d.line(18, d.get_y(), 192, d.get_y())
d.ln(7)
d.set_font("DJV", "", 9.6)
d.set_text_color(*INK)
d.multi_cell(W, 5.4,
    "Written for a reader with no background in EEG. Every technical term is defined the first "
    "time it appears. Nothing here is simulated: every number comes from a measurement made on "
    "these recordings.")
d.ln(8)
d.set_font("DJVM", "", 9)
d.set_text_color(*GREY)
for line in ["Subjects      2 (each recorded at every site)",
             "Sessions      4 per subject",
             "Sites         Sheffield · Leeds · Manchester",
             "Trials        6,718 with a measured timing error",
             "Author        Sourojit Goswami"]:
    d.multi_cell(W, 5.2, line)

# ============================ 1 ============================
d.add_page()
h1("1.  The problem, in plain language")
p("An EEG experiment works by showing a person something — a picture, a sound, a flash — and "
  "recording the electrical activity of their brain at the same time. To learn anything, you have "
  "to line the two up: you need to know exactly when the picture appeared, so you can look at the "
  "brain activity that followed it.")
p("The way this is normally done is that the computer running the experiment writes a timestamp "
  "into the recording at the moment it sends the picture to the screen. Everyone then treats that "
  "timestamp as the moment the picture appeared.")
p("It is not. Between the software deciding to show a picture and the light actually leaving the "
  "screen there is a chain of delays: the graphics card, the cable, the monitor's own internal "
  "processing, and the fact that a screen paints itself line by line rather than all at once. The "
  "timestamp is written at the start of that chain. The picture appears at the end of it.")
note("Nobody disputes that this delay exists. What is almost never done is measuring it — and "
     "because it is not measured, it is usually assumed to be small, constant, and therefore "
     "harmless. This work tests all three assumptions.")
p("We measured it directly, at three different laboratories, using the same two people at each "
  "one. The delay turned out to be between 24 and 135 milliseconds depending on the laboratory — "
  "large enough to move a brain response clean out of the time window a researcher would look for "
  "it in.")

h1("2.  A glossary, before anything else")
p("These terms are used throughout. Each is defined once here.")
term("EEG (electroencephalography)",
     "Recording the brain's electrical activity using metal sensors resting on the scalp. It is "
     "completely painless and measures voltages of a few millionths of a volt (microvolts, µV).")
term("Electrode / channel",
     "One sensor on the scalp, and the signal it produces. Our recordings use 14 sensors that all "
     "three laboratories had in common. Each is named for where it sits on the head — for example "
     "'Pz' is on the midline towards the back, 'PO7' and 'PO8' are lower and to either side.")
term("Reference",
     "A voltage is always a difference between two points, so every EEG signal is measured relative "
     "to some reference point. Different laboratories choose different reference points, which "
     "makes their recordings look different for reasons that have nothing to do with the brain. We "
     "recompute all of them relative to the same reference (the average of the two earlobes) so "
     "they can be compared.")
term("Trigger / marker / event",
     "The timestamp the experiment computer writes into the recording to say 'a stimulus was shown "
     "now'. This is the thing whose accuracy this whole project is about. 'Marker' and 'trigger' "
     "mean the same thing here.")
term("Trial",
     "One presentation of one stimulus, and the slice of brain recording that follows it. Our "
     "sessions contain a few hundred trials each.")
term("Epoch",
     "The slice of recording cut out around a single trial — here, from 200 milliseconds before "
     "the stimulus to 950 milliseconds after it. Cutting epochs is how continuous recording becomes "
     "a set of comparable trials.")
term("ERP (event-related potential)",
     "The brain response to a stimulus, isolated by averaging many trials together. A single trial "
     "is mostly noise; averaging 150 of them cancels the noise and leaves a repeatable waveform "
     "with characteristic bumps. Those bumps are named by their direction and timing — the 'N170' "
     "is a negative deflection about 170 milliseconds after a face is seen; the 'P3b' is a positive "
     "one from about 300 to 600 milliseconds after a rare, task-relevant stimulus.")
term("Latency",
     "When a bump in the ERP occurs, measured in milliseconds after the stimulus. This is what gets "
     "corrupted when the timestamp is wrong.")
term("Photodiode",
     "A small sensor that converts light into voltage. Taped to a corner of the screen and recorded "
     "on a spare channel of the same amplifier, it produces a sharp change the instant the screen "
     "actually changes. It is the ground truth in this study — the honest answer against which the "
     "computer's timestamp can be checked.")
term("Millisecond (ms)",
     "One thousandth of a second. For scale: ERP components are separated by tens of milliseconds, "
     "so an error of 75 ms is not a rounding issue — it is larger than the distance between two "
     "distinct brain responses.")
term("Offset versus jitter",
     "An offset is a delay that is the same on every trial (everything is late by 75 ms). Jitter is "
     "a delay that varies randomly from trial to trial. The distinction matters enormously: "
     "averaging many trials cancels jitter, but an offset survives averaging completely untouched.")
term("Variance",
     "A measure of how much a set of numbers spreads out. If you can split total variance into "
     "portions attributable to different causes, you learn which cause matters most.")
term("Artifact",
     "Non-brain electrical activity contaminating the recording — most often blinks, eye movements "
     "and muscle tension, which are far larger than genuine brain signals. Trials too contaminated "
     "to use are discarded.")
term("Filtering",
     "Removing frequency ranges that carry no useful signal — very slow drift (from sweat and "
     "electrode settling) and very fast activity (mostly muscle and electrical mains noise).")
term("Sampling rate",
     "How many times per second the amplifier measures the voltage. Different sites recorded at "
     "different rates, so all recordings are converted to a common 256 samples per second.")
term("BIDS",
     "A community standard for how brain-imaging files and their descriptions are laid out in "
     "folders, so that other researchers can read a dataset without instructions.")

# ============================ 3 ============================
d.add_page()
h1("3.  The dataset")
h2("3.1  Why the same people travelled to every site")
p("If you record different people at different laboratories and find a difference, you cannot tell "
  "whether it came from the equipment or from the people. This study avoids that by sending the "
  "same subjects to every site — a 'traveling subjects' design. Any difference that remains belongs "
  "to the laboratory.")
p("The recordings analysed here are two subjects (sub-SHE01 and sub-SHE02), each recorded in four "
  "sessions, in this order: Sheffield, then Leeds, then Sheffield again, then Manchester. The "
  "repeat visit to Sheffield is deliberate — it lets us separate 'this laboratory' from 'this "
  "particular day', because if the two Sheffield visits disagree, the cause cannot be the building.")
p("Each site used different equipment: a different amplifier, a different electrode cap and a "
  "different reference point. That is the point. The study is about what varies between real "
  "laboratories, not about a fault at any one of them.")

h2("3.2  Seven recordings were made per session. Two are used here.")
p("Each visit followed the same battery of tasks. Only two of them can support a timing "
  "measurement, and it is worth being explicit about why the other five cannot.")
for a, b in [
    ("n170 — USED", "The subject saw photographs of faces and of cars. Faces produce a strong, "
     "sharply-timed negative response about 170 ms later over the back of the head. Two balanced "
     "categories and a sharp landmark in time make this ideal."),
    ("p3b — USED", "The subject watched a stream of shapes and kept a mental count of rare target "
     "shapes among frequent standard ones. Rare, relevant stimuli produce a large positive response "
     "between about 300 and 600 ms. Again: two categories, clear timing."),
    ("n2pc — not used", "The subject attended to one side of the screen. There are visual stimuli, "
     "but the comparison of interest is left-versus-right (a difference in space), which does not "
     "give a clean landmark in time to measure a shift against."),
    ("resting — not used", "The subject sat quietly with eyes open and closed. There are no stimuli "
     "at all, so there are no timestamps that could be wrong."),
    ("blinks — not used", "The subject blinked on cue. Recorded specifically to characterise what "
     "blink contamination looks like, so it can be identified elsewhere. Not a brain response to a "
     "stimulus."),
    ("eog — not used", "The subject moved their eyes on cue, for the same purpose as the blink run."),
    ("assr — not used", "The subject heard an amplitude-modulated tone. This is an auditory "
     "recording, and a light sensor on the screen cannot measure when a sound arrived."),
]:
    d.set_font("DJV", "B", 9.7)
    d.set_text_color(*ACCENT)
    d.multi_cell(W, 4.8, a)
    d.set_font("DJV", "", 9.7)
    d.set_text_color(*INK)
    d.multi_cell(W, 4.8, b)
    d.ln(2.2)
note("Anticipated question: 'why not use the resting-state or blink data?' Because this study "
     "measures the gap between a recorded timestamp and a picture appearing on a screen. Recordings "
     "with no pictures, or with sounds instead of pictures, contain no such gap to measure. They "
     "are valuable for other parts of the project — they simply cannot answer this question.")

# ============================ 4 ============================
d.add_page()
h1("4.  How the timing error was measured")
p("This is the part that makes the study unusual, so it is worth being precise about it.")
p("During every session a photodiode was fixed to the screen and recorded on a spare channel of "
  "the same amplifier as the EEG. Because it is recorded by the same amplifier, its signal sits on "
  "exactly the same clock as the brain data and the timestamps — there is no need to synchronise "
  "two devices, which would introduce its own uncertainty.")
p("That gives two timestamps for every single trial:")
bullet("the timestamp the experiment computer wrote when it sent the stimulus, and")
bullet("the moment the photodiode saw the screen actually change.")
d.ln(1.4)
p("The dataset ships both, as two event tables covering the same recording. They have the same "
  "number of rows in the same order, describing the same events. So the timing error for a trial "
  "is a subtraction:")
d.ln(0.6)
d.set_font("DJVM", "", 10.2)
d.set_text_color(*ACCENT)
d.multi_cell(W, 5.6, "    error (ms)  =  (true onset  −  logged trigger) × 1000")
d.ln(3.2)
d.set_text_color(*INK)
p("No model. No estimate. No assumption about what the delay ought to be. It is measured on each "
  "of 6,718 trials individually.")
note("One technical detail, in case it is asked. The published dataset is split into one file per "
     "task, each with its own starting point, while the two event tables cover the whole session. "
     "Matching them up is done by finding the unique stretch of the session table whose sequence of "
     "trigger codes matches the task file exactly. The offset between the two turned out to be "
     "exactly constant across every visual event of a run — zero variation, verified in all 24 runs "
     "— which is what allows the original recording to be re-cut at the uncorrected timestamps "
     "without touching the multi-gigabyte source files.")
p("One honest caveat is carried through all the reporting. At Manchester, the correction procedure "
  "also applied a 3 ms shift to every timestamp in the recording, including button presses that "
  "have no visual component at all. That part is a global clock adjustment rather than a "
  "screen-delay measurement. The headline numbers report the error as-is, because that is what a "
  "laboratory without a photodiode would genuinely be wrong by; a separate column removes the "
  "global shift for anyone interpreting response latency specifically.")

h1("5.  Preprocessing: what was done to the recordings, and why")
p("Before anything can be compared, recordings made on different equipment have to be brought onto "
  "common ground. Every step below is applied identically to both versions of the data.")
h2("5.1  Harmonising the three sites")
bullet("Keep only the 14 sensors that all three laboratories had in common, so no comparison "
       "depends on a sensor one site did not have.")
bullet("Recompute every signal relative to the same reference point — the average of the two "
       "earlobes — cancelling the effect of each site's different reference choice.")
bullet("Filter to keep only 0.5 to 30 cycles per second, removing slow drift below and muscle and "
       "mains noise above.")
bullet("Convert all recordings to 256 samples per second, so one sample means the same duration "
       "everywhere.")
d.ln(1.2)
note("Order matters here: the recordings are resampled before filtering, because designing this "
     "particular filter directly at the original high sampling rate is numerically unstable and "
     "produces garbage.")
h2("5.2  Cutting the trials — twice")
p("Each trial is cut out from −200 ms to +950 ms around the stimulus. The 200 ms before the "
  "stimulus is the 'baseline': its average voltage is subtracted from the whole trial, so every "
  "trial starts from the same zero and slow drifts between trials cannot masquerade as a response.")
p("Then the crucial step: every trial is cut twice from the identical recording — once at the "
  "photodiode-verified onset, and once at the timestamp the computer logged. This produces two "
  "versions of the same data that differ in one respect only, which is where the cut was made.")
h2("5.3  The two controls that make the comparison fair")
bullet("A trial is discarded if it contains a voltage swing larger than 250 µV on any sensor, which "
       "indicates a blink or movement rather than brain activity. This threshold is deliberately "
       "permissive: because it triggers on the worst sensor, a strict threshold would throw away "
       "around 70% of trials on account of frontal blinks while barely affecting the responses at "
       "the back of the head that these two tasks actually depend on. Between 86% and 98% of trials "
       "are retained, consistently across all three sites.")
bullet("A trial is kept only if it survives that check under BOTH cuts. If each version were "
       "cleaned independently, the two would end up built from different sets of trials, and any "
       "difference between them could be caused by which trials were kept rather than by timing. "
       "This single rule is what makes the comparison a clean test of timing alone.")

# ============================ 6 ============================
d.add_page()
h1("6.  What we found")
h2("6.1  The error is large, steady within a session, and different at every site")
p("The figure below plots every measured trial. Each dot is one trial: horizontally, when in the "
  "session it occurred; vertically, how late its timestamp was.")
figure("p1_raw_trials.png",
       "Every trial from both subjects across all four sessions. Leeds sits near 24 ms, Manchester "
       "near 76 ms, Sheffield between 70 and 135 ms. Within any one block of trials the spread is "
       "only a few milliseconds — the error is a steady offset, not random noise. The visible steps "
       "in the Sheffield sessions are changes between recording blocks.")
p("Two things are immediately visible. First, the size: even the best site is 24 ms late, and the "
  "worst is over 130 ms. Second, the shape: within a block the dots form a nearly flat line. This "
  "is an offset, not jitter — and that is bad news, because averaging trials together removes "
  "jitter but leaves an offset completely intact.")

h2("6.2  The error belongs to the laboratory, not to the trial")
p("If the error is not random, where does it come from? Splitting the total variability by level "
  "answers the practical question: to predict a trial's timing error, what do you most need to know?")
figure("p2_variance.png",
       "Share of total variance in the measured timing error attributable to each level. Site "
       "accounts for 77%, session 17%, subject 4%, and trial-to-trial variation only about 1%.",
       width=140)
p("The answer is: which building the recording was made in. Site accounts for 77% of all the "
  "variability, and the trial level — the part that averaging could remove — accounts for barely "
  "1%. To our knowledge no previous study has partitioned trigger timing error this way, and it is "
  "this decomposition, rather than the mere existence of a delay, that is the contribution.")
note("Why this matters for anyone pooling data across sites: a constant offset that differs between "
     "laboratories does not average away, and it is not visible in the data unless you measured it. "
     "It shows up as an apparent difference in brain response timing between sites — a finding "
     "about equipment masquerading as a finding about people.")

d.add_page()
h2("6.3  The brain response moves by exactly that amount, and keeps its shape")
p("The previous sections concern timestamps. The question that matters is what this does to the "
  "actual brain measurement. Re-cutting the identical recording at the uncorrected timestamps and "
  "re-averaging gives the answer directly.")
figure("p3_erp_shift.png",
       "Left: the shift in the averaged brain response, recovered from the EEG itself, plotted "
       "against the timing error measured independently by the photodiode. Points fall on the "
       "diagonal — the two agree to 1.21 ms on average across 16 comparisons. Right: the "
       "correlation between the corrected and uncorrected waveforms, once realigned, stays above "
       "0.97 in every comparison.")
p("The left panel is a genuine independent check. The photodiode says the timestamps were late by "
  "a certain amount; the brain data, analysed without reference to the photodiode, says the "
  "response moved by that same amount. They agree to about one millisecond. That agreement "
  "validates the whole processing chain before any conclusion is drawn from it.")
p("The right panel says the response is not damaged — realign it and it is the same waveform, with "
  "correlation above 0.97 everywhere. This is exactly what section 6.2 predicts: if the error were "
  "random from trial to trial, averaging would blur the response into a smeared, lower-amplitude "
  "version of itself. It does not. It arrives late, intact.")

h2("6.4  The brain responses themselves")
p("The scatter above summarises sixteen comparisons as sixteen dots. It is worth also seeing the "
  "actual averaged brain responses those dots came from, because the effect is visible directly.")
figure("p5_erp_waveforms.png",
       "Averaged brain responses, pooled by site. Solid lines are cut at the true onset; dashed "
       "lines are the identical data cut at the timestamp the computer logged. The shaded band is "
       "the time window in which a laboratory would measure the component. Traces are smoothed by "
       "24 ms for display only — the smoothing is symmetric, so it cannot move anything in time.")
p("What is plotted is a 'difference wave': the average response to one condition minus the average "
  "response to the other. Subtracting the two conditions removes everything the brain was doing "
  "that was not specific to the comparison of interest, leaving the component itself.")
p("In the P3B panel on the right, the effect is unmistakable. The solid Sheffield trace rises "
  "inside the shaded measurement window. The dashed Sheffield trace — the same trials, the same "
  "recording, only cut at the uncorrected timestamp — rises noticeably later, with a substantial "
  "part of it pushed past the right-hand edge of the window. The N170 panel on the left is noisier, "
  "because it comes from only two subjects and the N170 is a much smaller response than the P3b, "
  "but the same displacement between solid and dashed traces is present.")
note("This picture is what the numbers in the previous sections mean in practice. A researcher does "
     "not see both traces — they only ever see one. If their laboratory does not correct the "
     "timing, they see the dashed one, and nothing in it announces that it is late.")

h2("6.4  The consequence: a laboratory reads the wrong number")
p("A researcher does not look at the whole waveform; they measure the response inside a standard "
  "time window — for the N170, roughly 130 to 200 ms. If the response has been displaced by 75 ms, "
  "it is no longer in that window when the measurement is taken.")
figure("p4_lab_reports.png",
       "The component amplitude a laboratory would report, with and without photodiode correction, "
       "for each of the 16 session-by-task comparisons. The measurement changes substantially, and "
       "in 4 of the 16 it comes out with the opposite sign.")
p("In 4 of the 16 comparisons the reported amplitude comes out with the opposite sign — not merely "
  "a weaker version of the correct result, but a qualitatively different one. This is the practical "
  "cost: without measuring your own timing, you may report a number whose sign is wrong, with no "
  "indication in the data that anything went awry.")

h1("7.  What it costs: decoding accuracy")
p("The sections above concern measurement — the number a researcher would write down. A second "
  "experiment asks a harder question: can a machine-learning classifier still tell the two "
  "conditions apart when the timing is wrong?")
h2("7.1  What AUC means")
p("The classifier is shown a single trial and asked which condition it came from. Its performance "
  "is scored with AUC — the area under the ROC curve. The number is easy to interpret even if the "
  "curve behind it is not:")
bullet("AUC = 0.5 means the classifier is guessing. It is the coin-flip line.")
bullet("AUC = 1.0 means perfect separation: every trial classified correctly.")
bullet("Values in between scale sensibly — AUC is the probability that a randomly chosen trial of "
       "one condition is scored higher than a randomly chosen trial of the other.")
d.ln(1.0)
p("AUC is used instead of plain accuracy because it does not depend on where you set the decision "
  "threshold, and it is not fooled when one condition occurs more often than the other — which is "
  "exactly the situation in the p3b task, where targets are deliberately rare.")
h2("7.2  The experiment")
p("Everything happens within a single session, so no site or hardware difference can contaminate "
  "the result. The trials of one session are split into two disjoint halves:")
bullet("The classifier is TRAINED on one half, using photodiode-corrected timing.")
bullet("It is then tested twice on the SAME held-out trials: once with corrected timing (the "
       "ceiling — the best the classifier could possibly do) and once with the uncorrected "
       "timestamps (what a laboratory without a photodiode would actually get).")
d.ln(1.0)
p("Because the two tests use identical trials from an identical recording, any difference between "
  "them is caused by timing and nothing else. The number of training trials is varied from 10 per "
  "condition up to all available, because the amount of data available is the other thing that "
  "matters in practice.")
figure("p6_auc.png",
       "Decoding performance as training data increases. Blue: correct timing, the best achievable. "
       "Red: the same trials with uncorrected timestamps. Green: uncorrected timestamps plus an "
       "augmentation that deliberately teaches the classifier that timing varies. The shaded gap "
       "between blue and red is the cost of bad timing.")
h2("7.3  What it shows")
p("For the N170 task with all trials available, correct timing reaches an AUC of 0.87 — a strong "
  "classifier. The identical data with uncorrected timestamps reaches 0.48, which is fractionally "
  "below the coin-flip line. The entire discriminative signal has been destroyed.")
p("There is a detail here that is worth pausing on, because it is counter-intuitive. As more "
  "training data is added, the corrected classifier steadily improves — and the uncorrected one "
  "gets slightly WORSE, drifting from 0.52 down to 0.48. More data does not help, because the "
  "classifier is learning the pattern more and more confidently at a position where it will not be "
  "at test time. Additional data makes it more certain about the wrong thing.")
p("The green line is a partial remedy. It shows what happens if, during training, trials are "
  "deliberately shifted back and forth in time so the classifier learns not to depend on the exact "
  "arrival time. This recovers roughly half of the lost performance — up to 0.69 for the N170 — but "
  "it does not restore the ceiling, and it comes at a price: a classifier taught to ignore timing "
  "also performs worse on correctly-timed data, because timing genuinely is informative when it is "
  "accurate.")
note("The honest summary of this experiment: augmentation is damage limitation, not a fix. Nothing "
     "recovers what a photodiode would simply have given you for free.")

h1("8.  What this does and does not show")
h2("Established")
bullet("Stimulus timing error at these three laboratories ranges from about 24 to 135 ms.")
bullet("It is a per-session constant, not trial-to-trial jitter: site accounts for 77% of the "
       "variance, trial about 1%.")
bullet("It displaces the brain response by exactly that amount while leaving its shape intact — "
       "confirmed against the photodiode to about 1 ms.")
bullet("It changes the number a laboratory would report, including its sign in 4 of 16 cases.")
bullet("It destroys decoding: AUC falls from 0.87 to 0.48 on the N170 task, and adding more "
       "training data makes the uncorrected classifier slightly worse rather than better.")
bullet("Shift augmentation recovers about half the lost decoding performance, but lowers "
       "performance on correctly-timed data — it is damage limitation, not a repair.")
h2("Limits to state plainly")
bullet("Two subjects at three sites. The traveling-subjects design makes the site comparison "
       "internally clean, but these are three specific laboratories, not a survey of the field.")
bullet("The delays measured are properties of the equipment as configured on those days. They are "
       "evidence that the error is large and site-specific, not a lookup table for other labs.")
bullet("The Manchester recordings carry an additional 3 ms global clock adjustment, reported "
       "separately from the screen-delay measurement.")
h2("The practical recommendation")
p("Measure your own timing rather than assuming it. A photodiode costs very little and, recorded on "
  "a spare amplifier channel, turns an unknown into a measured quantity. Where a study pools data "
  "across sites, this measurement should be reported alongside the results — because on this "
  "evidence, differences in equipment timing between laboratories are large enough to be mistaken "
  "for differences in brains.")

d.output(OUT)
print("wrote", OUT, "%.1f KB" % (os.path.getsize(OUT) / 1024))
