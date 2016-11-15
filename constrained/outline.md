## Infinite-stage set-up 
*Ref: Constructing Dynamic Treatment Regimes in Infinite-Horizon Settings. Ashkan Ertefaie*

### Dataset
Observed dataset 
$$D = \{  (X_{i,0}, A_{i,0}, R_{i,0} , X_{i,1}, A_{i,1}, R_{i,1}  \cdots, 
X_{i,T_i-1} , A_{i,T_i-1}, R_{i,T_i-1}, X_{i,T_i}) \}_{i=1}^n$$

 $X_{i,t}$ :  a set of covariates collected at the $t$-th decision point 
 
$A_{i,t}$ :  treatment assigned at the time point after measuring $X_{i,t}$ , which is referred as *action* in RL

$R_{i,t}$ : a reward received after treatment $A_{i,t}$ is assigned. $R^+$ / $R^-$, positive/negative reward; $R$, general reward.

$T_i$ : the number of time point for the i-th trajectory

$S_{i,t} = (X_{i,t-k}, A_{i,t-k},X_{i,t-k+1}, A_{i,t-k+1}, \cdots, X_{i,t})$ : the summary of the observed history till time point $t$, including the history of covariates and  treatments.  Assumed to depend on, at most, the last $k$ time points. It is also referred as *state variable* in RL

$\gamma$, discount factor

Assume that the support of $S_t$ is the same for every time point, denoted as $\mathcal{S}$. Given $S_t =s$, $A_t$ takes values in $\mathcal{A}_{s} = \{ 0, 1, 2, \cdots, m_s \}$ for all $t$ where $m_s < \infty$. If a patient dies before the last time point, say at decision point $t$, we set $S_t = \emptyset$, which is a absorbing state.  For $s = \emptyset$, we set $A_s = \emptyset$. The histories of treatment and the summary function through $t$ are denoted by $\bar{A}_t$ and $\bar{S}_t$, respectively

A treatment regime (*policy*), $\pi$, is a deterministic decision rule, which maps the support of the summary variable $S$ to the support of the possible treatment assginment. That is, $\pi : \mathcal{S} \to \mathcal{A}_s$, for each $s \in \mathcal{S}$.

### Assumptions

* Potential outcomes  assumptions

Let $S^*_{t+1}(\bar{a}_t)$ denote the potential outcome of the summary variable at the $(t+1)$-th decision point, if the individual had been following the treatment history $\bar{a}_{t}$. The following potential outcome assumptions are assumed.

 Consistency: $S^*_{t+1} (\bar{A}_t) = S_{t+1}$ for each $t$
		
Sequential randomization : $\{S^*_{t+1}(\bar{a}_t),S^*_{t+2}(\bar{a}_{t+1}), \cdots, S^*_{T}(\bar{a}_{T-1}) \} \indep  A_t \mid \bar{S}_t, \bar{A}_{t-1} = \bar{a}_{t-1}$
		
Positivity assumption: Let $p_{A|S}(a|s)$ be the conditional probability of receiving treatment $a$ given $S=s$. For each action $a \in \mathcal{A}_s$ and for every possible  value $s$, $P_{A | S}(a | s) > 0$

* Markovian assumptions

Markovian assumption: For each $t$,

$S_t \indep \bar{S}_{t-2}, \bar{A}_{t-2} \mid S_{t-1}, A_{t-1}$
 
$A_t \indep \bar{S}_{t-1}, \bar{A}_{t-1} \mid S_{t}$

Time homogeneity: For each $s \in \mathcal{S}$ and $a \in \mathcal{A}_s$
$p(S_{t+1} \in \mathcal{B} \mid S_t = s, A_t =a) = p(S^{\prime} \in \mathcal{B} \mid S = s, A =a$, where $S$ and $S^{\prime}$ are the summary function at the current and the next time, respectively. 


## Define a constrained optimal regime

### Goal
Goal is to construct a treatment regime that, if implemented, would maximize, on average over the state, the value for positive rewards, while satisfying the constraint on the values for negative rewards. Specifically, $$V^{\pi}_{R^{+}} = \int Q_{R^{+}}^{\pi}(s, \pi(s)) \,d G(s),$$
 where $G(s)$ denotes the distribution of the states. Same applies to $V^{\pi}_{R^{-}}$. 
 
 A constrained optimal regime in infinite-stage is  a solution to 
$$\underset{\pi}{\text{maximize }}  V_{R^{+}}^{\pi}$$
$$\text{subject to }   V_{R^{-}}^{\pi} \le \kappa.$$

### Policy search 

We focus on linear decision rule and define the class of interest as $ \Pi = \{ \pi: \pi(s) = Interval\{\tau^{\intercal} \phi(s)  \}, s \in \mathcal{S}, \tau \in \mathbb{R}^{dim(s)} \}$, where $\phi(s)$ is a vector of feature function of state $s$. Policy search over $\boldsymbol{\tau}$.


### Value functions

The state-action value function at time $t$, is an expected value of the cumulative discounted reward, if taking treatment $a$ at state $s$ at time $t$ and following the policy $\pi$ afterward. Under the Markovian assumption, the state-action value function does not depend on $t$. Thus, denote it by $Q^{\pi}(s,a)$, which is bounded, when $\gamma < 1$. $Q_t^{\pi}(\emptyset, a) =0$.

$Q^{\pi}(s, a) = \mathbb{E}_{\pi}\left[\sum_{k=1}^{\infty} \gamma^{k-1} R_{t+k} | S_t =s, A_t=a \right] = \mathbb{E}_{\pi} \left[R_{t+1} + \gamma Q^{\pi}(S_{t+1}, \pi(S_{t+1})) | S_t =s, A_t=a \right]$
	
The last  equation is known as Bellman equation for $Q^{\pi}(s,a)$. 
 The state value function following policy $\pi$ is defined as

$$V_{R}^{\pi}(s) = Q_{R}^{\pi}(s,\pi(s))  = \mathbb{E}_{\pi} \left[ \sum_{k=1}^{\infty}\gamma^{k-1}R_{t+k} | S_t=s, A_t =\pi(s) \right].$$


### Estimate Q functions

Impossible to visit all the states infinitely many times in medical application, especially when the state variables are continuous. Need to interpolate from visited states to unvisited states.

Linear architectures, where Q is approximated by a linear parametric combination of k basis functions (features $\phi_j$): $$\widehat{Q}^{\pi}(s, a; \omega) = \sum_{j=1}^{k} \phi_j(s,a)\omega_j$$

**LSTDq in Least-square policy iteration**

Linear architectures, where Q is approximated by a linear parametric combination of k basis functions (features $\phi_j$): $$\widehat{Q}^{\pi}(s, a; \omega) = \sum_{j=1}^{k} \phi_j(s,a)\widehat{\omega}_j$$
 
LSQ step in LSPI: Least-Squares Fixed-Point Approximation
 
1. Force the approximate Q function to be a fixed point under the Bellman operator: $T\widehat{Q}^{\pi} \approx  \widehat{Q}^{\pi}$.

Bellman operator $T$: $TQ^{\pi}(s,a) = R(s,a) + \gamma \sum_{ s' \in \mathcal{S}} P(s,a,s')\sum_{ a' \in \mathcal{A}} \pi(a',s')Q(s',a')$. Bellman residual minimizing approximation is another choice.

2. A sample $(s, a, r, s^{\prime})$ contributes to the approximation:
			$$\widehat{A} \leftarrow \widehat{A} + \phi(s,a)\{ \phi(s,a)^\intercal - \gamma\phi(s^{\prime},\pi(s^{\prime}))^\intercal \},$$
			 $$\widehat{b} \leftarrow \widehat{b} + \phi(s,a)r$$
3. Solve the linear system for $\omega^{\pi}$, 
			$$ A\omega^{\pi} = b$$
			
LSPI is completed by choosing the policy $\pi(s^{\prime}) = max_a \widehat{Q}(s^{\prime},a;\widehat{\omega})$.
	
## Simulation

### Data simulation

*Ref: Reinforcement learning design for cancer clinical trialsYufan Zhao, Michael R. Kosorok, and Donglin Zeng*


* A system of ODE model: 

$W_t$ : patient wellness (toxicity)

$M_t$ : tumor size

$D_t$ : chemotherapy agent dose (dose is the action $A_t$ here)

$$\dot{W}_t = a_1 \text{max }(M_t, M_0) + b_1 (D_t - d_1) $$

$$\dot{M}_t = [a_2 \text{min }(W_t, W_0) - b_2 (D_t - d_2)] \times 1 (M_t >0)$$
where decision points are $t= 0, 1,\cdots, T-1$, and $T=6$. $\dot{W}_t$ and $\dot{M}_t$ are the transition functions. These changing rate yields a piece-wise linear model over time.  Constants value are set as 
$a_1 = 0.1, a_2 = 0.15, b_1 = 1.2, b_2 = 1.2, d_1 = 0.5$ and $d_2 = 0.5$.  The initial state variables and the dose actions for each decision points are draw as follow
$$M_0 \sim \text{Uniform}(0, 2)$$
$$W_0 \sim \text{Uniform}(0, 2)$$
$$D_0 \sim \text{Uniform}(0.5, 1)$$ 
$$D_t\sim \text{Uniform}(0.5, 1), t = 1, \cdots, 5$$

The state variables can be obtained via:
$$W_{t+1} = \text{max } (W_t + \dot{W}_t, 0)$$
$$M_{t+1} = \text{max } (M_t + \dot{M}_t, 0)$$
Tumor size $M_t$ min is zero. Toxicity $W_t$ min is zero as well.

* The survival indicator $F_t$:

Assume everyone is alive at the initial decision point $t=0$, $p_0 =0$, $F_0 = 0$.

Death events occur during time interval $(t -1, t]$, $t = 1, 2, \cdots, 6$, and are recorded at the end of each interval as variable $F_t$, $t=1,2, \cdots, 6$.

Assume that survival status depends on both toxicity and tumor size. For each time interval $(t − 1, t], t = 1,..., 6$, define the hazard function as $λ(t)$, which satisfies
$$\text{log } \lambda(t) = \mu_0 + \mu_1 W_t + \mu_2 M_t$$
where I picked $\mu_1 = \mu_2 = 1$, and $\mu_0 = -8.5$ (not mentioned in paper). This again is a piece-wise linear approximation.

$$\lambda(t) = \text{exp}\{ \mu_0 + \mu_1 W_t + \mu_2 M_t\}$$

The cumulative hazard function during time interval $(t-1, t]$ is

 \\[ \Delta \Lambda(t) = \int_{t-1}^t \lambda(s) \,ds
= \int_{t-1}^t exp\{ \mu_0 + \mu_1 W_t + \mu_2 M_t\} \,ds
=\text{exp}\{ \mu_0 + \mu_1 W_t + \mu_2 M_t\} \\]

The survival function
$$\Delta F(t) = \text{exp} [ - \Delta \Lambda(t)]$$
$$ = \text{exp}[ - \text{exp}\{ \mu_0 + \mu_1 W_t + \mu_2 M_t\} ]$$

The random event of death during time interval $(t-1, t], \, F_t , \, t = 1, \cdots, 6$,
$$F_t \sim \text{Bernoulli}(p_t) $$
$$p_t = 1 - \Delta F(t) = 1 - \text{exp}[ - \text{exp}\{ \mu_0 + \mu_1 W_t + \mu_2 M_t\} ]$$
If $F_{t-1} = 1$, then $F_t =1$. Also, if death occurred, all the other state variables at the following decision points are all set to nan. 

* Rewards
Reward is a bivariate vector, consisting of positive reward and negative reward.
$r_t = R_t(s_t, a_t, s_{t+1}), \,  \boldsymbol{R}_t = ( R^+_t , R^-_t )$

Positive rewards :
Positive reward is used to assess tumor size reduction

Negative rewards :
Negative reward is to assess the increase of patient negative wellness / toxicity

  If death $F_{t+1} = 1$   | O.W.
  -------------------  | -------------
  $R_t^- = 60$       | $R_t^- = 5 \,(W_{t+1} - W_t \ge -0.5); 0 \,(O.W)$
  $R_t^+ = 0$        | $R_t^+= 15 \,(M_{t+1} = 0) ; 5 \, (M_{t+1} - M_t \le -0.5 \ M_{t+1} \ne 0); 0 \,(O.W.)$

#### Overall 
The trajectories / training data generated according to the ODE model 
$$\{S_{0i}, A_{0i}, R_{0i}, S_{1i},\cdots,S_{5i}, A_{5i}, R_{5i}, S_{6i} \}_{i=1}^N$$
where action is the does level $A_t = D_t$, $S_t = (M_t, W_t, F_t)$, discount factor $\gamma =0.8$.

There are 7 decision points. The last point has only states $(M_6, W_6)$, but no action either reward.

$$S_0 \overset{D_0}{\longrightarrow}\underset{\underset{R_0}{\curvearrowright}}\, S_1 \overset{D_1}{\longrightarrow} \underset{\underset{R_1}{\curvearrowright}}\, S_2 \overset{D_2}{\longrightarrow} \underset{\underset{R_2}{\curvearrowright}}\,  S_3 \overset{D_3}{\longrightarrow} \underset{\underset{R_3}{\curvearrowright}}\,  S_4 \overset{D_4}{\longrightarrow}  \underset{\underset{R_4}{\curvearrowright}}\, S_5 \overset{D_5}{\longrightarrow} \underset{\underset{R_5}{\curvearrowright}}\,  S_6 $$
where $S_t = (M_t, W_t, F_t)$, $t = 0, 1, \cdots, 6$. Moreover, $R_{t} = (R_t^+, R_t^-), t = 0, 1, \cdots, 5$.

Note action $[0,1] \to 0, 0.2, 0.4, 0.6, 0.8, 1$

Note $t = 1, \cdots, 7$ in code.

## Discretization of the state space and the action space

#### Discrete the state space 

Create a grid to discretize the state space $\mathcal{S}$ 

Choice of grid generation

1. Uniform grids [X]
2. Quadrature grids
3. Sobol's / Tezuka grids : low discrepancy
4. Random grids

Choice of approximation

1. Deterministic transition onto nearest vertices 0'th order approximation [X]
2. Stochastic transition onto neighboring vertices 1'st order approximation: Kuhn triangulation	

#### Discretize the time space

Discretize time in a variable way such that one discrete time transition roughly corresponds to a transition into neighboring grid points/regions. Time is fixed as month in data generation, but doesn't have to be.

*CFL condition: Courant Friedrichs Levy*
In the two-dimensional case, the CFL condition becomes
$$C =  \frac{\mu_x \Delta t}{\Delta x} + \frac{\mu_y \Delta t}{\Delta y}  \le C_{max}$$
The value of $C_{\max }$ changes with the method used to solve the discretized equation, especially depending on whether the method is explicit or implicit. If an explicit (time-marching) solver is used then typically $C_{\max }=1$. Implicit (matrix) solvers are usually less sensitive to numerical instability and so larger values of $C_{\max }$ may be tolerated.  Explicit method only uses current time, while implicit method uses current and next time.

[x] extremely inefficient for solution of stationary problems unless local time-stepping i.e. $\Delta t=\Delta t(x)$ is employed

#### Discretize the action space
$[0,1] \to 0, 0.2, 0.4, 0.6, 0.8, 1$ 5 levels of dose


## Radial basis function construction
Idea is the inseparable more likely to be separable in higher dimension space.


Basis function construction $1, 2, \cdots, K$ with diagonal structure and K+1 independent parameters
$$X = (M, W)$$
$$\phi_k ( x) = \text{exp}\left[ - \frac{\| x - \mu_k \|}{2\sigma^2_k} \right]$$

* A set of features for positive rewards
$$\phi_k^{+}( x) = \text{exp}\left[ - \frac{\| x - \mu_k \|}{2\sigma^2_k} \right]$$
* A set of features for negative rewards
$$\phi_k^{-}( x) = \text{exp}\left[ - \frac{\| x - \mu_k \|}{2\sigma^2_k} \right]$$
### Pick parameters $c_k$ and $\mu_k$
Ways to pick parameters

1. The simplest approach is to randomly select a number of training examples as RBF centers. This method has the advantage of being very fast, but the network will likely require an excessive number of centers. Once the center positions have been selected, the spread parameters $\sigma_k$ can be estimated, for instance, from the average distance between neighboring centers. [X]

2. Unsupervised selection: clustering, density estimation
 
3. Cross-validation based on Bellman residuals or $T\widehat{Q}- \widehat{Q}$ ?? Need to think about the criterion???

4. Supervised ....


##  Value function / action value function estimation

1. LSTDq
2. Stochastic gradient descent
3. TD methods: Q-learning


## Policy search in the class

We focus on linear decision rule and define the class of interest as 
$\Pi = \{ \pi \}$

$\pi(s) = \mathbb{I}(  \tau^{\intercal} \phi(s)  < 0.8 ) = 1$

$\pi(s) = \mathbb{I}( 0.6 < \tau^{\intercal} \phi(s) < 0.8 ) = 0.8$

$\pi(s) = \mathbb{I}( 0.4 < \tau^{\intercal} \phi(s) < 0.6 )  = 0.6$

$\pi(s) = \mathbb{I}( 0.2 < \tau^{\intercal} \phi(s) < 0.4 ) = 0.4$

$\pi(s) = \mathbb{I}( 0 < \tau^{\intercal} \phi(s) < 0.2 ) = 0.2$

$\pi(s) = \mathbb{I}( \tau^{\intercal} \phi(s) < 0  )= 0$

How do I define the class of policy?


## Existence of stationary policy under constraints

Discretization is applied here to make sure the optimal constrained exist, and will need to assess how close it is to the true optimal?

Mean? Continuous time? Reparameterization trick



