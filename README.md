# Portfolio and Its End-to-End Combinatorial Optimization Strategy
## Preface
Given the large number of formulas in this project, we recommend reading this README.md in a local Markdown editor to prevent display issues on platforms like GitHub. Next, we will introduce how to run the project's source code:
(i) Unzip the compressed package `main.zip`;
(ii) Directly run the file at the path `\rethink_exp\test programing.py`;
(iii)Select the baseline method as prompted to complete the model training, validation and testing operations.


## Problem Introduction

**Problem Background:** Asset allocation stands as a pivotal mechanism in facilitating the circulation of capital across diverse sectors of the economy, thereby playing a crucial role in enhancing overall economic efficiency. In the realm of financial portfolio management, the optimization of asset distribution aims to strike a balance between maximizing returns and mitigating risks, a challenge that has garnered significant attention in both academic research and practical applications.

**Prediction Phase:** The predictive task involves leveraging historical features, such as daily price series and trading volume data, to forecast the daily return $y$ for a set of $N$ stocks. Mathematically, let $\mathbf{x}_i$ denote the feature vector associated with stock $i$, which may include, but is not limited to, 10-day, weekly, monthly, and annual historical returns, as well as rolling averages over these time windows. The goal is to predict the return $\hat{y}_i$ for stock $i$ on the subsequent trading day, as below.
$\hat{y}_i = \mathcal{M}_\theta (\mathbf{x}_i)$

**Decision Phase:** The optimization objective is to maximize the expected portfolio return while simultaneously minimizing the associated risk. We define the following variables and parameters:
- $\mathbf{v} \in \mathbb{R}^N$: A vector where $v^i$ represents the fraction of capital invested in stock $i$, with $0\leq v^i\leq 1$ and $\sum_{i = 1}^{N} v^i = 1$.
- $\mathbf{y} \in \mathbb{R}^N$: A vector of expected returns for each stock, where $y^i$ is the predicted return for stock $i$.
- $\lambda = 0.1$: A risk-aversion parameter that quantifies the trade-off between return and risk.
- $\mathbf{Q} \in \mathbb{R}^{N\times N}$: A positive semi-definite matrix that characterizes the covariance structure between the returns of different stocks.

The optimization problem is formulated as below.
$\mathbf{v}^*(\mathbf{y})=\arg\max_{\mathbf{v}} \mathbf{v}^\top\mathbf{y}-\lambda\mathbf{v}^\top\mathbf{Q}\mathbf{v}$
$\text{subject to: }\sum_{i = 0}^{N} v^i = 1$

Here, the first term $\mathbf{v}^\top\mathbf{y}$ represents the expected return of the portfolio, and the second term $\lambda\mathbf{v}^\top\mathbf{Q}\mathbf{v}$ accounts for the portfolio risk, where the covariance matrix $\mathbf{Q}$ captures the interdependencies between stock returns.

**Dataset and License:** The dataset employed in this study is sourced from the publicly available SP500 dataset: Quandl (2022), which contains financial data of 505 of the largest companies in the US market spanning from 2004 to 2017. The feature set for each stock includes historical returns over multiple time horizons (10-day, weekly, monthly, and annual) and rolling averages computed over these periods. In our experimental setup, we set the risk-aversion parameter $\lambda = 0.1$.

The publicly accessible dataset utilized in this research is obtained from the specified website. Users must adhere to the terms and conditions outlined in the corresponding data usage agreement for its legitimate utilization.

## Baseline Comparison Methods Introduction
**SPO**
In parallel, an alternative research strand has concentrated on adapting subgradient approximation methodologies, originally devised for continuous linear problems, to discrete-valued scenarios. Specifically, the SPO-relax method introduces a relaxation of the original discrete optimization problem and leverages the surrogate SPO+ loss function, first proposed in  Mandi et al. (2020). This loss formulation enables the utilization of subgradient-based updates within a backpropagation-compatible paradigm. Mathematically, the SPO-relax loss is defined as below.

$
\mathcal{L}_{\text{spo}}(\mathbf{y},\hat{\mathbf{y}}) = -f\bigl(\mathbf{v}^*(2\hat{\mathbf{y}}-\mathbf{y}),2\hat{\mathbf{y}}-\mathbf{y}\bigr) + 2f\bigl(\mathbf{v}^*(\mathbf{y}),\mathbf{y}\bigr) - f\bigl(\mathbf{v}^*(\mathbf{y}),\mathbf{y}\bigr)
$

**NCE**
Mandi et al. (2022) t take $ \mathbb{S} \setminus \{\mathbf{v}^*(c)\} $ as negative examples and define a noise-contrastive estimation (NCE) loss, as below.


$
\mathcal{L}_{\text{NCE}}(\hat{c}, c)=\frac{1}{|\mathbb{S}|}\sum_{\mathbf{v} \in \mathbb{S}}\left(f(\mathbf{v}^*(c), \hat{\mathbf{c}}) - f(\mathbf{v}, \hat{\mathbf{c}})\right)
$

The novelty lies in the above formula being differentiable without solving the optimization problem. Moreover, if solutions in $ \mathbb{S}$ are optimal for arbitrary cost vectors, this approach is equivalent to training within a region of the convex hull of $\mathbb{V}$.

**CpLayer**
Agrawal et al. (2019)  propose an approach to differentiate through disciplined convex programs (a subset of convex optimization problems used in domain-specific languages). Introducing disciplined parametrized programming (a subset of disciplined convex programming), they show every such program can be represented as composing an affine map from parameters to problem data, a solver, and an affine map from solver solution to original problem solution.

**Identity**
Sahoo et al. (2023)  propose a hyperparameter-free approach to embed discrete solvers as differentiable layers in deep learning. Prior methods (input perturbations, relaxation, etc.) have drawbacks like extra hyperparameters or compromised performance. Their work leverages the geometry of discrete solution spaces, treats solvers as negative identities in backpropagation, and uses generic regularization to avoid cost collapse. \textbf{I} is the identity matrix, and the gradient designed in their paper is shown as below.

$
\frac{\partial \mathbf{v}}{\partial \mathbf{y}} = -\mathbf{I}
$

**LODL and DFL**
Mandi et al. (2022) propose a novel approach that abandons surrogates entirely, instead learning loss functions tailored to task-specific information. Notably, theirs is the first method to fully replace the optimization component in decision-focused learning with an automatically learned loss. Key advantages include: (a) reliance only on a black-box oracle for solving the optimization problem, ensuring generalizability; (b) convexity by design, enabling straightforward optimization.

**Blackbox**
When confronted with the dilemma that the map from $\mathbb{C}\rightarrow\mathbb{V}$ is either non-differentiable or has vanishing gradients,  Poganˇci´c et al. (2019)  adopt a remarkably straightforward remedy: they approximate the gradient via linear interpolation. Their surrogate gradient construction is shown as below:
$
\frac{\partial \mathcal{L}}{\partial \mathbf{y}} = \frac{1}{\lambda} \left[ \mathbf{v} \left( \hat{\mathbf{y}} + \lambda \frac{\partial L}{\partial \mathbf{v}} (\hat{\mathbf{v}}) \right) - \mathbf{v} (\hat{\mathbf{y}}) \right]
$

**2-Stage**
To ensure an equitable comparison, all end-to-end trainable models and the 2-stage baseline share an identical predictive backbone: a compact multi-layer perceptron (MLP).
Given an input feature vector $\mathbf{x}$, the predictor $\mathcal{M}$ is defined by the recursive relation:

$
\mathbf{a}^{(1)} = \mathbf{x}
$

$
\mathbf{a}^{(i+1)} = \phi\!\bigl(\mathbf{W}^{(i)}\mathbf{a}^{(i)}+\mathbf{b}^{(i)}\bigr),  i=1,\dots,K-1
$

$
\hat{\mathbf{y}} = \mathbf{a}^{(K)}
$

where $\mathbf{W}^{(i)}$ and $\mathbf{b}^{(i)}$ denote the weight matrix and bias vector of the $i$-th layer, respectively, and $\phi(\cdot)=\max(\cdot,0)$ is the ReLU activation.
Throughout the experiments we fix the depth at $K=3$ and the hidden dimension at $32$.

The 2-stage paradigm serves as the standard baseline whenever the coefficients of the downstream optimization task are uncertain and must be forecast.
A supervised predictor is trained on the pre-collected dataset $\mathcal{D}=\{(\mathbf{c}_i,\mathbf{y}_i)\}_{i=1}^{N}$ to minimize either the mean square error (MSE) loss as below.

$
\mathcal{L}_{\text{MSE}}(\hat{\mathbf{y}},\mathbf{y}) = \frac{1}{N}\sum_{i=1}^{N}\|\mathbf{y}_i-\hat{\mathbf{y}}_i\|^2
$

or the binary cross-entropy (BCE) loss as below:

$
\mathcal{L}_{\text{BCE}}(\hat{\mathbf{y}},\mathbf{y}) = -\frac{1}{N}\sum_{i=1}^{N}\Bigl[y_i\log\hat{y}_i+(1-y_i)\log(1-\hat{y}_i)\Bigr]
$

At test time, the inferred coefficients $\hat{\mathbf{c}}=\mathcal{M}_\theta(\mathbf{x})$ are treated as deterministic inputs, after which an off-the-shelf solver is invoked to obtain the final decision.

Notably, the overall training objective in this 2-stage pipeline is entirely dictated by the \emph{prediction} loss (MSE or BCE); no task-specific decision loss is backpropagated.

## Experiments Results
Evaluation Metric: Regret. Regret is defined as follows. We hope that for a set of regret values (c, ĉ), the value is zero, or preferably, as small as possible. A smaller value indicates that the benefit of the post-prediction decision is closer to the benefit of the prior decision.
$regret (\mathbf{c},\hat{\mathbf{c}})=||f(\mathbf{z}^*(\mathbf{c}); \mathbf{c})-f(\mathbf{z}^*(\hat{\mathbf{c}}); \mathbf{c})||
$

#### Results on the test set:
| Methods         | Relative Regret on  Portfolio |
|-----------------|-----------|
| 2-stage         | 0.243     |
| DFL             | 0.380     |
| Blackbox        | 0.286     |
| Identity        | 0.280     |
| CPLayer         | 0.309     |
| SPO             | 0.245     |
| LODL            | **0.160** |
| NCE             | 0.367     |
| Org-LTR         | 0.214     |
| SAA-LTR (ours)  | 0.333     |


