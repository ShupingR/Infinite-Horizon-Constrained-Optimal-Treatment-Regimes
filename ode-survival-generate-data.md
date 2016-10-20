Time interval $(t -1, t]$, $t = 1, 2, \cdots,6$

Log of the hazard function
$$log \lambda(t) = \mu_0 + \mu_1 W_t + \mu_2 M_t$$
where $\mu_1 = \mu_2 = 1$

$$\lambda(t) = exp\{ \mu_0 + \mu_1 W_t + \mu_2 M_t\}$$

The cumulative hazard function
$$ \triangle \Lambda(t) = \int_{t-1}^t \lambda(s) \,ds$$
$$= \int_{t-1}^t exp\{ \mu_0 + \mu_1 W_t + \mu_2 M_t\} \,ds$$
$$=exp\{ \mu_0 + \mu_1 W_t + \mu_2 M_t\}$$

The survival function
$$\triangle F(t) = exp [ - \triangle \Lambda(t)]$$
$$ = exp[ - exp\{ \mu_0 + \mu_1 W_t + \mu_2 M_t\} ]$$

The random event of death = 1, $\text{Bernoulli}(p) $
$$p = 1 - \triangle F(t) = 1 -exp[ - exp\{ \mu_0 + \mu_1 W_t + \mu_2 M_t\} ]$$


A constrained optimal regime in infinite-sta
