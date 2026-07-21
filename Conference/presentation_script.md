# Presentation Script — KMGPR Conference Talk (15 minutes)

> **Total time:** 15 minutes (12 slides + Q&A buffer)
> **Pacing rule:** ~1–1.5 min per slide, except Title & Thank You

---

## Slide 1 — Title Slide (~30s)

**On screen:** Paper title, authors, affiliation (HUST)

**Script:**

> Good morning/afternoon, everyone. My name is [Tên], from Hanoi University of Science and Technology. Today, I will present our work titled *"A K-Means-based Gaussian Process Regression Model for Human Motion Prediction in Human–Robot Collaboration."*

**→ Transition:** *Click to next slide*

---

## Slide 2 — Table of Contents (~20s)

**On screen:** I. Introduction, II. Proposed Method, III. Validation Example, IV. Conclusion

**Script:**

> The presentation is organized into four main sections. First, I will introduce the background and motivation. Then, I will describe the proposed method. After that, I will present the validation results. And finally, the conclusion and future work.

**→ Transition:** *Click to next slide*

---

## Slide 3 — Background and Motivation (~1.5 min)

**On screen:** 3 bullets + Fig. 1 (HRC illustration)

**Script:**

> In human-robot collaboration, the robot must anticipate human motion to avoid collisions and improve task efficiency. As illustrated in this figure, collaborative tasks such as co-carrying, handover, and co-assembling all require the robot to predict human motion in advance.

> However, human motion is inherently nonlinear and stochastic, which makes accurate prediction challenging. Furthermore, predictions without confidence estimates may cause overconfident robot behavior. In safety-critical applications like physical human-robot interaction, uncertainty quantification is essential — it allows the robot to adopt conservative behaviors when prediction confidence is low.

> *(Point to Fig. 1)* This motivates our central question: **How can robots predict human motion while assessing the uncertainty of the predictions?**

**→ Transition:** *"Let me now discuss the current research gap."*

---

## Slide 4 — Research Gap & Main Contributions (~2 min)

**On screen:** Research Gap (3 bullets) + Main Contributions (3 bullets)

**Script:**

> Looking at the existing literature, we identify three key limitations.

> First, model-based methods rely on explicit physical models, which makes them struggle with the nonlinear and stochastic nature of human motion.

> Second, deep learning approaches such as RNN, LSTM, and GRU can capture temporal dependencies, but they require large datasets and, importantly, they lack built-in uncertainty quantification mechanisms.

> Third, standard Gaussian Process Regression provides uncertainty estimates through its predictive variance, but it suffers from O(N³) computational complexity due to the N-by-N matrix inversion — making it infeasible for real-time applications.

> *(Pause briefly)*

> To address these limitations, we make three main contributions.

> First, we propose a K-Means-based GPR model — KMGPR — that reduces computational cost while maintaining prediction accuracy comparable to standard GPR.

> Second, we demonstrate uncertainty quantification through predictive variance, validated on two representative HRC scenarios.

> And third, we provide a theoretical analysis explaining the prediction stability differences between KMGPR and standard GPR under abrupt motion changes.

**→ Transition:** *"Now, let me introduce the proposed method."*

---

## Slide 5 — Multi-dimensional GP Architecture (~1.5 min)

**On screen:** Problem Formulation (Input/Output) + For each dimension (mean/variance equations) + Red O(N³) highlight + Fig. 2

**Script:**

> We formulate human motion prediction as a regression problem. Given an observation window of h past time steps — represented as the input X_t in R^{dh} — the goal is to predict the next position Y_t in R^d.

> *(Point to Fig. 2)* Since the motion is d-dimensional, we decompose the problem into d separate dimensions, each modeled by an independent Gaussian Process. Each GP takes the h-step window as input and outputs a Gaussian predictive distribution — giving us both the predicted position through the mean, and the prediction confidence through the variance.

> *(Point to equations)* For each dimension, the predictive mean and variance are computed using these standard GPR equations.

> *(Point to red text)* However, notice that both equations require inverting an N-by-N covariance matrix — this is where the **O(N³) computational complexity** comes from. For our dataset with over 2400 training windows, this becomes a significant bottleneck for real-time prediction.

**→ Transition:** *"To overcome this bottleneck, we propose the KMGPR framework."*

---

## Slide 6 — Proposed KMGPR Framework (~1.5 min)

**On screen:** Training phase + Prediction phase (2 bullets) + Fig. 3

**Script:**

> The key idea of KMGPR is simple: instead of training on the full dataset, we use K-Means clustering to select a small but representative subset.

> *(Point to Fig. 3, Training Phase)* In the **training phase**, the Elbow method is used to determine the optimal number of clusters. K-Means then partitions the full training data and selects the data point closest to each centroid as the representative sample. This gives us a subset of size M, which is much smaller than the original N.

> *(Point to Fig. 3, Prediction Phase)* In the **prediction phase**, replacing the full training dataset with this representative subset reduces the covariance matrix from N-by-N to M-by-M, bringing the time complexity down from O(N³) to O(M³).

> In our experiments, M equals 484 — only 20% of the original 2424 training windows — yet the model maintains comparable prediction accuracy, as I will show in the validation results.

**→ Transition:** *"Let me now present the validation setup."*

---

## Slide 7 — Dataset Description (~1 min)

**On screen:** 3 bullets + Fig. 4 (data distribution) + Fig. 5 (Elbow curve)

**Script:**

> Our training data consists of 30 trajectories from the initial position toward three distinct targets, generated within 10 seconds at a sampling interval of 0.1 seconds.

> 80% of these trajectories — that is, 24 trajectories — were used for training, generating a total of 2424 sliding windows.

> *(Point to Fig. 5)* Applying the Elbow method, we identified the optimal number of clusters at M = 484.

> *(Point to Fig. 4)* As you can see, despite retaining only 20% of the original data, the subset effectively preserves the spatial coverage of the full dataset — demonstrating the robustness of K-Means in extracting representative samples.

**→ Transition:** *"Next, let me show the training configuration."*

---

## Slide 8 — Validation Setup / Training Parameters (~1 min)

**On screen:** Table I (6 parameters) + Objective function: Log-marginal likelihood with data-fit and complexity penalty annotations

**Script:**

> Table I summarizes the key parameters used for model training. We used a window size of 10 time steps, the Squared Exponential kernel with Automatic Relevance Determination, and the Adam optimizer with an initial learning rate of 0.05 over 250 iterations.

> *(Point to equation below table)* The hyperparameters are optimized by maximizing the log-marginal likelihood. I want to draw your attention to two important terms in this objective function: the **data-fit term**, which measures how well the model fits the training data, and the **complexity penalty**, which penalizes overly complex models. The balance between these two terms is governed by the size of the training dataset — and this will become important when we analyze the results of Scenario 2.

**→ Transition:** *"Now let's look at the results."*

---

## Slide 9 — Scenario 1: Nominal Trajectories (~1.5 min)

**On screen:** Table II (MSE + Prediction Time) + Fig. 6 (9 subplots with Target 1/2/3 labels)

**Script:**

> In Scenario 1, both models are evaluated on nominal trajectories toward three learned targets.

> *(Point to Table II)* As shown in Table II, KMGPR achieves an overall MSE of 0.21 times 10⁻⁶ square meters, while standard GPR achieves 0.19. The difference is negligible — both models maintain nearly equivalent prediction accuracy.

> However, the key advantage is in **computational efficiency**: KMGPR completes prediction in 4.62 seconds, compared to 41.09 seconds for standard GPR — a reduction by approximately a factor of eight.

> *(Point to Fig. 6)* The figure on the right shows the prediction performance of KMGPR with 95% confidence intervals. Despite using only 20% of the original training data, the model maintains consistently narrow confidence intervals across all three targets, confirming that the extracted subset still provides sufficient information for high-confidence predictions.

**→ Transition:** *"Now let's examine a more challenging scenario."*

---

## Slide 10 — Scenario 2: Abrupt Motion Changes (~2 min)

**On screen:** 2 key observation bullets + Fig. 7 (KMGPR vs GPR predictions) + Fig. 8 (Length scale comparison)

**Script:**

> Scenario 2 evaluates the models on an atypical trajectory with abrupt motion changes. The human operator first accelerates, then stops abruptly at t = 4 seconds.

> *(Point to Fig. 7, top row)* After the abrupt stop, both models initially overshoot — which is expected since such sudden deceleration patterns are absent from the training data. However, their recovery behaviors are markedly different.

> KMGPR returns to the stationary position along a smooth, stable trajectory — without oscillation.

> *(Point to Fig. 7, bottom row)* In contrast, standard GPR exhibits a highly oscillatory trajectory, deviating significantly before eventually converging.

> *(Point to Fig. 8)* **Why does this happen?** The answer lies in the length scales. As shown in this comparison, KMGPR consistently learns **larger** length scales across all input features, while standard GPR learns smaller ones.

> Remember the log-marginal likelihood we saw earlier. With the full dataset, the data-fit term dominates, pushing the optimization toward smaller length scales to fit every data point closely. But with the reduced subset from K-Means, the pressure on the data-fit term is lower, so the optimization converges at larger length scales.

> Larger length scales produce a smoother predictive function — which is why KMGPR recovers stably. Conversely, the smaller length scales of standard GPR cause sharp variations, leading to the oscillatory predictions we observed.

**→ Transition:** *"Let me now summarize our findings."*

---

## Slide 11 — Conclusion (~1 min)

**On screen:** Summary (3 bullets) + Limitations (2 bullets) + Future Works (2 bullets)

**Script:**

> To summarize, KMGPR maintains prediction accuracy comparable to standard GPR while significantly reducing computational cost — by approximately a factor of eight.

> The model effectively captures predictive uncertainty through the variance of Gaussian Process Regression.

> And under abrupt motion changes, KMGPR exhibits greater prediction stability with significantly fewer oscillations compared to standard GPR.

> Regarding limitations, the model has been validated only on two-dimensional simulated data, and the subset selection via K-Means is performed offline and remains fixed after training.

> For future work, we plan to validate the model with real-world physical human-robot interaction in a three-dimensional framework, and to develop adaptive control schemes that leverage the predicted uncertainty for real-time decision-making.

**→ Transition:** *Click to final slide*

---

## Slide 12 — Thank You (~10s)

**On screen:** HUST logo + "Thank You!"

**Script:**

> Thank you for your attention. I am happy to take any questions.

---

# Q&A Preparation

Below are anticipated questions and suggested responses:

### Q1: "Why not use random sampling instead of K-Means?"
> Random sampling does not guarantee that the selected subset preserves the spatial distribution of the training data. K-Means ensures that each region of the input space is represented by the data point closest to its centroid, which maintains data coverage while reducing redundancy.

### Q2: "Can this extend to 3D?"
> Yes. Since each dimension is modeled by an independent GP, extending to 3D simply adds one more GP to the architecture. The K-Means subset selection operates on the input space and is dimension-agnostic.

### Q3: "What about the computational cost of K-Means itself?"
> K-Means is performed offline during the training phase, so it does not affect real-time inference. Its time complexity is O(N·M·I), where I is the number of iterations — much lower than the O(N³) of standard GPR.

### Q4: "Why use Elbow method instead of other methods like Silhouette?"
> The Elbow method provides a straightforward criterion based on the SSE curve and does not require computing pairwise distances between all samples, making it more computationally efficient for our use case. However, exploring alternative selection criteria is a valid direction for future work.

### Q5: "Why does the MSE of KMGPR increase slightly compared to standard GPR?"
> The slight increase is expected — we are using only 20% of the training data. The subset cannot capture every fine-grained pattern in the full dataset. However, the difference is negligible (0.21 vs 0.19 × 10⁻⁶ m²), and the trade-off in computational efficiency is substantial.

### Q6: "Can you explain more about the data-fit term and complexity penalty?"
> *(This is a backup answer in case someone asks about the length scale analysis in detail)*
> In the log-marginal likelihood, the data-fit term measures how well the model explains the observed data, while the complexity penalty discourages overfitting by penalizing overly flexible models. With a large dataset, the data-fit term dominates — pushing the model toward smaller length scales to closely fit each data point. With the reduced K-Means subset, the data-fit pressure is lower, allowing the optimization to converge at larger length scales, which correspond to smoother predictive functions.

---

# Timing Summary

| Slide | Content | Time |
|---|---|---|
| 1 | Title | 0:30 |
| 2 | Table of Contents | 0:20 |
| 3 | Background & Motivation | 1:30 |
| 4 | Research Gap & Contributions | 2:00 |
| 5 | Multi-dim GP Architecture | 1:30 |
| 6 | KMGPR Framework | 1:30 |
| 7 | Dataset Description | 1:00 |
| 8 | Validation Setup | 1:00 |
| 9 | Scenario 1: Nominal | 1:30 |
| 10 | Scenario 2: Abrupt Motion | 2:00 |
| 11 | Conclusion | 1:00 |
| 12 | Thank You | 0:10 |
| | **Total** | **~14:00** |

> [!TIP]
> Remaining ~1 minute serves as buffer for natural pauses, audience reactions, or minor time adjustments.
