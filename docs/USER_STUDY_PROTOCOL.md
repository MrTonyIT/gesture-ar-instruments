# User Study Protocol: Ergonomic & Latency Evaluation of Gesture AR Instruments

**Document Version:** 1.0.0  
**Target System:** `gesture-ar-instruments` (`research-upgrade-v2`)  
**Methodology:** Controlled Laboratory Within-Subjects Experiment ($N = 24$)

---

## 1. Research Objectives & Hypotheses

This protocol defines a rigorous, reproducible evaluation procedure to assess the usability, ergonomic comfort, and interaction fidelity of spatial AR musical instruments.

### 1.1 Hypotheses
* **$H_1$ (Jitter & Precision):** The adaptive 1€ filter will yield significantly lower subjective tremor and higher note selection accuracy than the raw baseline during fine positioning tasks.
* **$H_2$ (Ergonomic Fatigue):** Desk-surface docking (wrists resting on the tabletop) will produce significantly lower physical demand scores on the NASA-TLX compared to floating mid-air AR instruments.
* **$H_3$ (Note Trigger False Positives):** Downward velocity gating ($v_y \ge 60\text{ px/s}$) and hover suppression will reduce unintended note re-strikes by at least $80\%$ compared to static geometric penetration testing.
* **$H_4$ (Chord Transition Efficiency):** Direct touch and finger-count chord selection on the 9-chord AR guitar will enable novice users to execute common chord progressions ($C \to G \to Am \to F$) with $< 300\text{ ms}$ inter-chord latency.

---

## 2. Participant Cohort & Apparatus

### 2.1 Demographics
* **Sample Size:** A target cohort of $N = 24$ participants (12 novices, 12 experienced musicians) is proposed for initial study planning. Before participant recruitment, a formal *a-priori* power analysis must be verified using the final statistical model, expected effect size, within-subject correlation, multiplicity strategy, and pilot variance estimates.
  - *Illustrative planning scenario:* In a repeated-measures within-subjects design ($4 \text{ Filter} \times 2 \text{ Posture}$ conditions), an assumed medium effect size (Cohen's $f = 0.25$, $\eta_p^2 = 0.06$) with moderate within-subject correlation ($\rho = 0.50$) at $\alpha = 0.05$ typically suggests $N \ge 24$ for conventional target power ($1 - \beta \ge 0.80$). If pilot testing indicates smaller effect sizes ($f < 0.20$), $N$ must be adjusted upward prior to formal recruitment.
* **Stratification:**
  * Group A (Novices, $n = 12$): No prior guitar or piano performance experience.
  * Group B (Musicians, $n = 12$): $\ge 2$ years active instrumental practice.
* **Inclusion Criteria:** Normal or corrected-to-normal vision and hearing; unconstrained mobility of both upper extremities.

### 2.2 Apparatus & Environment
* **Workstation:** Standard desktop PC (60 Hz display, resolution 1920x1080).
* **Sensor:** Fixed monocular 1080p RGB webcam positioned at eye level (tilt angle $-15^\circ$ toward desk surface).
* **Illumination:** Standard indoor diffuse office lighting (400–600 lux).
* **Audio:** Stereo headphones with known hardware latency.
* **Desk Surface:** Matte, non-reflective desk surface providing resting forearm support.

---

## 3. Experimental Design

A **within-subjects factorial design** ($4 \text{ Filter Modes} \times 2 \text{ Posture Conditions}$) counterbalanced via balanced Latin squares to mitigate learning and fatigue order effects.

```
+-----------------------------------------------------------------------------+
|                          Within-Subjects Conditions                         |
+------------------------------+----------------------------------------------+
| Filter Condition             | Posture Condition                            |
| 1. Raw Pass-Through          | A. Desk-Docked (wrists on physical desk)    |
| 2. EMA Filter (alpha=0.65)   | B. Floating Mid-Air (unsupported arms)       |
| 3. Adaptive Deadband         |                                              |
| 4. One Euro Filter (Casiez)  |                                              |
+------------------------------+----------------------------------------------+
```

---

## 4. Standardized Task Battery

### Task 1: Static Reticle Stability (Jitter Assessment)
* **Instructions:** Align the right index fingertip with a target marker on the screen and hold steady for $10\text{ seconds}$.
* **Repetitions:** 3 trials per filter condition.
* **Metrics:** RMS displacement in pixels from the centroid; maximum peak-to-peak deviation.

### Task 2: C-Major Scale Cadence (Piano Kinematics)
* **Instructions:** Play an 8-note ascending C-major scale ($C4 \to D4 \to E4 \to F4 \to G4 \to A4 \to B4 \to C5$) at a target tempo of 60 BPM (guided by metronome).
* **Repetitions:** 5 trials per condition.
* **Metrics:**
  * Timing error relative to metronome beat (ms).
  * Double-strike false positive count (unintended extra triggers).
  * Missed key count (downward velocity insufficient to trigger).

### Task 3: 4-Chord Pop Progression (Guitar Strumming)
* **Instructions:** Perform a standard 4-chord progression: $[C] \to [G] \to [Am] \to [F]$, strumming each chord on downbeats 1 and 3 of a $4/4$ measure at 68 BPM.
* **Repetitions:** 4 continuous cycles per trial.
* **Metrics:**
  * Chord transition delay (time between last strum of chord $k$ and chord switch confirmation of $k+1$).
  * Plucked string count per strum (verifying unmuted strings sound and muted strings remain silent).
  * Strum volume consistency (velocity standard deviation).

---

## 5. Evaluation Metrics & Instruments

### 5.1 Objective Metrics (Required Study Instrumentation)
> [!NOTE]
> These objective metrics are specified for automated data logging in future experimental study instrumentation. They are not currently persisted by default in the core application codebase and must be recorded by the experimental study test harness during participant sessions:
1. **Timestamp:** High-resolution monotonic clock (`time.perf_counter()`).
2. **End-to-End Latency:** Time from frame capture arrival to audio sample buffer dispatch.
3. **Tracking Error:** Spatial divergence between raw landmark detection and filtered state.
4. **Trigger Velocity:** Measured vertical pixel velocity $v_y$ at key strike.
5. **False Trigger Count:** Note onsets without a deliberate downward velocity stroke.

### 5.2 Subjective Instruments
Administered immediately following each experimental block:
1. **NASA Task Load Index (NASA-TLX):**
   * Mental Demand (0–100)
   * Physical Demand (0–100)
   * Temporal Demand (0–100)
   * Performance (0–100)
   * Effort (0–100)
   * Frustration (0–100)
2. **System Usability Scale (SUS):** Standard 10-item questionnaire.
3. **Perceived Latency Likert Scale:** 7-point scale ranging from $1$ ("Noticeably delayed / sluggish") to $7$ ("Instantaneous / perfectly responsive").
4. **Physical Comfort Rating:** 5-point fatigue assessment of shoulders, forearms, and wrists.

---

## 6. Statistical Analysis Plan

* **Distribution Assessment:** Shapiro-Wilk test for normality and Mauchly’s test for sphericity across all repeated-measures levels. Where sphericity is violated ($\epsilon < 0.75$), Greenhouse-Geisser corrections are applied; otherwise Huynh-Feldt corrections.
* **Continuous Metrics (Tracking Error, Jitter RMS, Timing Error):**
  - Analyzed using Linear Mixed-Effects Models (LMM) with `Filter` (4 levels) and `Posture` (2 levels) as within-subjects fixed factors, `Musical Expertise` (Novice vs Musician) as a between-subjects fixed factor, and `Participant` as a random intercept:
    $$\text{Metric} \sim \text{Filter} \times \text{Posture} \times \text{Expertise} + (1 \mid \text{Participant})$$
  - Effect sizes reported as partial eta-squared ($\eta_p^2$) for omnibus tests and Cohen’s $d_z$ (with 95% bootstrap confidence intervals) for pairwise contrasts.
* **Ordinal & Survey Instruments (NASA-TLX, SUS, Likert):**
  - Evaluated using Aligned Rank Transform (ART) ANOVA or non-parametric Wilcoxon signed-rank tests for paired within-subjects comparisons.
* **Multiplicity Correction:**
  - All post-hoc pairwise comparisons are adjusted using the Holm-Bonferroni step-down procedure or Benjamini-Hochberg False Discovery Rate (FDR) control at $q = 0.05$.
* **Hypothesis Evaluation Criteria:**
  - $H_1$ (Jitter Reduction): Evaluated by the contrast between 1€ and Raw filter on static jitter RMS. Supported if 1€ yields a statistically significant reduction ($p_{\text{adj}} < 0.01$) with a large effect size ($d_z > 0.80$, 95% CI excluded from 0).
  - $H_2$ (Ergonomic Fatigue): Evaluated on the NASA-TLX Physical Demand subscale. Supported if Desk-Docked posture produces significantly lower scores than Floating Mid-Air ($p_{\text{adj}} < 0.001$, $d_z \ge 0.80$, with 95% CI).
  - $H_3$ (False Note Suppression): Evaluated by unintended strike count on the piano. Supported if downward velocity gating reduces false positives by $\ge 80\%$ compared to static geometric penetration ($p_{\text{adj}} < 0.01, \eta_p^2 > 0.14$).
  - $H_4$ (Chord Transition Efficiency): Evaluated by inter-chord transition delay. Supported if novice participants achieve mean transition latencies $< 300\text{ ms}$ ($95\%\text{ CI} \subset [0, 300]\text{ ms}$) across consecutive strum bars.
