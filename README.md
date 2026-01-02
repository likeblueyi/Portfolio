# SCHEDULING (ENERGY) and Its End-to-End Combinatorial Optimization Strategy
## Problem Introduction
**Problem Background:**  
With the increasing penetration of clean energy sources into the power grid, energy demand profiles and pricing mechanisms have exhibited enhanced adaptability. Within the realm of industrial production, the optimization of scheduling tasks with respect to real-time energy prices holds significant potential for substantial energy conservation and operational cost reduction. This study focuses on formulating and solving an energy-cost-aware scheduling problem, which encompasses both predictive and optimization components.

**Prediction Phase:**  
The predictive task entails forecasting the energy price for each of the 48 time slots (each slot corresponding to a 30-minute interval) within a given scheduling horizon. To achieve this, a set of relevant features is utilized, including, but not limited to, weather predictions (such as temperature forecasts), wind energy production estimates, and other operational parameters (e.g., production load forecasts). Mathematically, for a time slot index \(t\), let \(\mathbf{x}_t\) denote the feature vector comprising these relevant attributes, and the goal is to predict the energy price \(p_t\), as below.
$$
\hat{p}_t = \mathcal{M}(\mathbf{x}_t)
$$

**Decision Phase:**  
The optimization objective is to minimize the total energy cost associated with scheduling \(J\) jobs on \(M\) machines, while adhering to the constraints of earliest start time \(e_j\) and latest end time \(l_j\) for each job \(j\).

We define the following parameters:
- \(M\): Number of machines available for job processing.
- \(J\): Number of jobs to be scheduled.
- \(R\): Number of resources required for job execution.
- \(T\): Number of time slots in a scheduling day (set to \(T = 48\) in this study, corresponding to 30-minute intervals).
- \(e_j\): Earliest start time of job \(j\).
- \(l_j\): Latest end time of job \(j\).
- \(d_j\): Duration of job \(j\).
- \(p_j^t\): Power usage of job \(j\) at time slot \(t\).
- \(u_{jr}\): Resource usage of job \(j\) on resource \(r\).
- \(c_{mr}\): Capacity of machine \(m\) for resource \(r\).

Let \(v^{jmt}\) be a binary decision variable, where \(v^{jmt}=1\) if job \(j\) starts at time slot \(t\) on machine \(m\), and \(v^{jmt} = 0\) otherwise. The objective function aims to minimize the total energy cost of the schedule, which is formulated as a linear program as below.
$$
\min_{\mathbf{v}} \sum_{j \in J} \sum_{m \in M} \sum_{t \in T} v^{jmt} \left( \sum_{t'=t}^{t + d_j - 1} p_j^{t'} \right)
$$

subject to the following constraints:

1. **Job Scheduling Uniqueness Constraint**: Each job is scheduled on exactly one machine at a unique start time, as shown below.
$$
\sum_{m \in M} \sum_{t \in T} v^{jmt}=1, \quad \forall j \in J \notin \mathcal{T}_{jm}
$$

2. **Machine-Job Compatibility Constraint**: A job cannot be scheduled on a machine outside the set of available machines for that job, as shown below.
$$
v^{jmt}=0, \quad \forall j \in J, \forall m \in M, \forall t \notin \mathcal{T}_{jm}
$$
where \(\mathcal{T}_{jm}\) represents the set of valid start times for job \(j\) on machine \(m\).

3. **Time Window Constraint**: The job must start after the earliest start time and end before the latest end time, as shown below.
$$
v^{jmt}=0, \quad \forall j \in J, \forall m \in M, \forall t + d_j>l_j
$$

4. **Resource Capacity Constraint**: The resource usage of all jobs scheduled on a machine must not exceed the machine's resource capacity, as shown below.
$$
\sum_{j \in J} \sum_{t'=t - d_j + 1}^{t} v^{jmt} u_{jr} \leq c_{mr}, \quad \forall m \in M, \forall r \in R, \forall t \in T
$$

In our experimental setup, we adopt \(N = 3\) machines, \(R = 1\) resource, and the resource usage \(u_{jr}\) is assumed to be known and constant.

**Dataset and License:**  
The dataset utilized in this study is sourced from the open-sourced Irish Single Electricity Market Operator (SEMO) dataset. This dataset contains energy-related data collected from midnight 1st November 2011 to 31st December 2013. The energy price prediction task at each time slot is based on a 9-dimensional feature vector, which includes:
- Calendar attributes (e.g., day of the week, month).
- Day-ahead weather characteristic estimates (such as wind speed, temperature forecasts).
- SEMO day-ahead forecasted energy load.
- Wind energy production and price forecasts.
- Actual measurements (including wind speed, temperature, \(\text{CO}_2\) intensity, and real-time price).

The publicly available SEMO dataset  adopted in this research is licensed and regulated by the Commission for Regulation of Utilities (CRU) in Ireland and the Utility Regulator for Northern Ireland (URENI, formerly known as NIAUR). Researchers utilizing this dataset must adhere to the licensing terms and regulatory requirements set forth by these authorities.

## Baseline Comparison Methods Introduction
**SPO**
In parallel, an alternative research strand has concentrated on adapting subgradient approximation methodologies, originally devised for continuous linear problems, to discrete-valued scenarios. Specifically, the SPO-relax method introduces a relaxation of the original discrete optimization problem and leverages the surrogate SPO+ loss function, first proposed in  Mandi et al. (2020). This loss formulation enables the utilization of subgradient-based updates within a backpropagation-compatible paradigm. Mathematically, the SPO-relax loss is defined as below.
$$
\mathcal{L}_{\text{spo}}(\mathbf{y},\hat{\mathbf{y}}) = -f\bigl(\mathbf{v}^*(2\hat{\mathbf{y}}-\mathbf{y}),2\hat{\mathbf{y}}-\mathbf{y}\bigr) + 2f\bigl(\mathbf{v}^*(\mathbf{y}),\mathbf{y}\bigr) - f\bigl(\mathbf{v}^*(\mathbf{y}),\mathbf{y}\bigr)
$$

**NCE**
Mandi et al. (2022) t take \( \mathbb{S} \setminus \{\mathbf{v}^*(c)\} \) as negative examples and define a noise-contrastive estimation (NCE) loss, as below.
$$
\mathcal{L}_{\text{NCE}}(\hat{c}, c)=\frac{1}{|\mathbb{S}|}\sum_{\mathbf{v} \in \mathbb{S}}\left(f(\mathbf{v}^*(c), \hat{\mathbf{c}}) - f(\mathbf{v}, \hat{\mathbf{c}})\right)
$$
The novelty lies in the above formula being differentiable without solving the optimization problem. Moreover, if solutions in \( \mathbb{S}\) are optimal for arbitrary cost vectors, this approach is equivalent to training within a region of the convex hull of \( \mathbb{V} \).

**CpLayer**
Agrawal et al. (2019)  propose an approach to differentiate through disciplined convex programs (a subset of convex optimization problems used in domain-specific languages). Introducing disciplined parametrized programming (a subset of disciplined convex programming), they show every such program can be represented as composing an affine map from parameters to problem data, a solver, and an affine map from solver solution to original problem solution.

**Identity**
Sahoo et al. (2023)  propose a hyperparameter-free approach to embed discrete solvers as differentiable layers in deep learning. Prior methods (input perturbations, relaxation, etc.) have drawbacks like extra hyperparameters or compromised performance. Their work leverages the geometry of discrete solution spaces, treats solvers as negative identities in backpropagation, and uses generic regularization to avoid cost collapse. \textbf{I} is the identity matrix, and the gradient designed in their paper is shown as below.
$$
\frac{\partial \mathbf{v}}{\partial \mathbf{y}} = -\mathbf{I}
$$

**LODL and DFL**
Mandi et al. (2022) propose a novel approach that abandons surrogates entirely, instead learning loss functions tailored to task-specific information. Notably, theirs is the first method to fully replace the optimization component in decision-focused learning with an automatically learned loss. Key advantages include: (a) reliance only on a black-box oracle for solving the optimization problem, ensuring generalizability; (b) convexity by design, enabling straightforward optimization.

**Blackbox**
When confronted with the dilemma that the map from $\mathbb{C}\rightarrow\mathbb{V}$ is either non-differentiable or has vanishing gradients,  Poganˇci´c et al. (2019)  adopt a remarkably straightforward remedy: they approximate the gradient via linear interpolation. Their surrogate gradient construction is shown as below:
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{y}} = \frac{1}{\lambda} \left[ \mathbf{v} \left( \hat{\mathbf{y}} + \lambda \frac{\partial L}{\partial \mathbf{v}} (\hat{\mathbf{v}}) \right) - \mathbf{v} (\hat{\mathbf{y}}) \right]
$$

**2-Stage**
To ensure an equitable comparison, all end-to-end trainable models and the 2-stage baseline share an identical predictive backbone: a compact multi-layer perceptron (MLP).
Given an input feature vector $\mathbf{x}$, the predictor $\mathcal{M}$ is defined by the recursive relation:
$$
\mathbf{a}^{(1)} = \mathbf{x}
$$
$$
\mathbf{a}^{(i+1)} = \phi\!\bigl(\mathbf{W}^{(i)}\mathbf{a}^{(i)}+\mathbf{b}^{(i)}\bigr),  i=1,\dots,K-1
$$
$$
\hat{\mathbf{y}} = \mathbf{a}^{(K)}
$$
where $\mathbf{W}^{(i)}$ and $\mathbf{b}^{(i)}$ denote the weight matrix and bias vector of the $i$-th layer, respectively, and $\phi(\cdot)=\max(\cdot,0)$ is the ReLU activation.
Throughout the experiments we fix the depth at $K=3$ and the hidden dimension at $32$.

The 2-stage paradigm serves as the standard baseline whenever the coefficients of the downstream optimization task are uncertain and must be forecast.
A supervised predictor is trained on the pre-collected dataset $\mathcal{D}=\{(\mathbf{c}_i,\mathbf{y}_i)\}_{i=1}^{N}$ to minimize either the mean square error (MSE) loss as below.
$$
\mathcal{L}_{\text{MSE}}(\hat{\mathbf{y}},\mathbf{y}) = \frac{1}{N}\sum_{i=1}^{N}\|\mathbf{y}_i-\hat{\mathbf{y}}_i\|^2
$$
or the binary cross-entropy (BCE) loss as below:
$$
\mathcal{L}_{\text{BCE}}(\hat{\mathbf{y}},\mathbf{y}) = -\frac{1}{N}\sum_{i=1}^{N}\Bigl[y_i\log\hat{y}_i+(1-y_i)\log(1-\hat{y}_i)\Bigr]
$$

At test time, the inferred coefficients $\hat{\mathbf{c}}=\mathcal{M}_\theta(\mathbf{x})$ are treated as deterministic inputs, after which an off-the-shelf solver is invoked to obtain the final decision.

Notably, the overall training objective in this 2-stage pipeline is entirely dictated by the \emph{prediction} loss (MSE or BCE); no task-specific decision loss is backpropagated.

## Experiments Results
Evaluation Metric: Regret. Regret is defined as follows. We hope that for a set of regret values (c, ĉ), the value is zero, or preferably, as small as possible. A smaller value indicates that the benefit of the post-prediction decision is closer to the benefit of the prior decision.
$regret (\mathbf{c},\hat{\mathbf{c}})=||f(\mathbf{z}^*(\mathbf{c}); \mathbf{c})-f(\mathbf{z}^*(\hat{\mathbf{c}}); \mathbf{c})||
$

#### Results on the test set:
| Methods         | Relative Regret on Scheduling (Energy) |
|-----------------|---------------------|
| 2-stage         | 1.793               |
| DFL             | 6.272               |
| Blackbox        | 6.503               |
| Identity        | 5.690               |
| CPLayer         | --                  |
| SPO             | $\textbf{\underline{1.505}}$ |
| LODL            | 1.786               |
| NCE             | 1.663               |
| Org-LTR         | 1.540               |
| SAA-LTR (ours)  | 2.339               |

