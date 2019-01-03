---
tags: Reinforcement Learning, theory, AI
---
# [Reinforcement Learning Lecture 4](https://www.youtube.com/watch?v=PnHCvfgC_ZA&list=PLweqsIcZJac7PfiyYMvYiHfOFPg9Um82B&index=4)
###### David Silver, Deep mind

## Model Free Prediction

While planning by dynamic programming  is about solving a known MDP, model-free prediction involves solving an environment that can be represented by an MDP but without the knowledge of the MDP. You do not know the dynamics of the environment. Model-free methods exploits the experience of the agent to estimate an value function.

Model free methods also have the two phases of _model-free prediction_ and _model-free control_. The goal of model-free prediction is to estimate the value function of an unknown MDP while the goals of model-free control is to optimize the value function of an unknown MDP.

### Monte-Carlo Reinforcement Learning

This is extremely effective and widely used in practice. 

- MC methods learn directly from episodes of experience
- without the knowledge of MDP transitions/rewards
- Looks at complete episodes: no bootstrapping
- Simplest possible idea for value function estimation is to compute the mean return of the episodic rewards
- Caveat: MC methods only works with episodic MDPs, hence all episodes must terminate

#### Monte-Carlo Policy Evaluation

- Goal: Given a policy $\pi$, learn $v_{\pi}$ from episodes of experience
    - $S_1, A_1, R_1, S_2, A_2, R_2,...S_T$ ~ $\pi$

- Revall that the return is the total discounted reward
    - $G_t = R_{t+1} + \gamma R_{t+1} + .... + \gamma^{T-1}R_T$

- Recall that the value function is the expected return
    - $v_{\pi}(s) = \mathrm{E}_{\pi}[G_t | S_t = s]$

- Monte-Carlo policy evaluation uses _empirical mean_ return instead of expected return

##### First-Visit MC Policy Evaluation

- Consider that there is some loop in your MDP and you always come to some state. To evaluate state $s$
- Over a _multiple episodes_, the **first** time-step $t$ that state is visited in an episode,
    - Increment counter $N(s) \leftarrow N(s) + 1$
    - Increment total return $S(s) \leftarrow S(s) + G_t$
    - Value is the estimate of mean return $V(s) = S(s)/N(s)$
    - By law of large numbers, that says the mean of a bunch of IID random variables does approach a true value, $V(s) \rightarrow v_{\pi}(s)$ as $N(s) \rightarrow \inf$
    - The only caveat is to visit the states as many times as possible

##### Every-Visit MC Policy Evaluation
- Over a _multiple episodes_, the **every** time-step $t$ that state is visited in an episode,
    - Increment counter $N(s) \leftarrow N(s) + 1$
    - Increment total return $S(s) \leftarrow S(s) + G_t$
    - Value is the estimate of mean return $V(s) = S(s)/N(s)$
    - Again, $V(s) \rightarrow v_{\pi}(s)$ as $N(s) \rightarrow \inf$
    - The only caveat is to visit the states as many times as possible

##### Incremental Mean

The mean $\mu_1,\mu_2,....$ of a sequence $x_1,x_2,...$ can be computed incrementally,

$\mu_k = \frac{1}{k} \sum_{j = 1}^k  x_j \Rightarrow \mu_{k-1} + \frac{1}{k}(x_k - \mu_{k-1})$

##### Incremental Monte-Carlo Updates

- Update $V(s)$ incrementally after an episode $S_1, A_1, R_2, ....S_T$
- For each state $S_t$ with return $G_t$
    - $N(S_t) \leftarrow N(S_t) + 1$
    - $V(S_t) \leftarrow V(S_t) + \frac{1}{N(S_t)}(G_t - V(S_t))$

- In non-stationary problems, it can be useful to track a running mean, i.e. forget old episodes using a fixed step size $\alpha$
    - $V(S_t) \leftarrow V(S_t) + \alpha (G_t - V(S_t))$

### Temporal-Difference Learning

- TD methods also lean directly from episodes of experience
- They do not have knowledge of MDP transitions/rewards
- TD learns from _incomplete_ episodes, by boot strapping i.e. uses an estimates of returns instead of actual return
- TD updates a guess towards a guess

#### MC and TD

- Goal: learn $v_{\pi}$ online from experience under policy $\pi$
- In incremental every-visit Monte-Carlo
    - we update the value $V(S_t)$, toward _actual_ return $G_t$
    - $V(S_t) \leftarrow V(S_t) + \alpha (G_t - V(S_t))$

- Simplest temporal-difference learning ($TD(0)$)
    - We update the value $V(S_t)$ toward _estimated_ return $R_{t+1} + \gamma V(S_t{t+1})$, that contains immediate reward plus estimated reward over rest of the trajectory
    - $V(S_t) \leftarrow V(S_t) + \alpha (R_{t+1} + \gamma V(S_t{t+1}) - V(S_t))$
    - $R_{t+1} + \gamma V(S_t{t+1})$ is called the $TD$ target, which is random and depends on what happens in the next time-step
    - $\delta_t = R_{t+1} + \gamma V(S_t{t+1}) - V(S_t)$ is called the $TD$ error

#### Advantages and Disadvantages of MC vs. TD

- TD can learn before knowing the final outcom
    - TD can learn online after every step
    - MC must wait until end of episode before return is known
- TD can learn without final outcome
    - TD can learn from incomplete sequences
    - MC can only learn from complete sequences
    - TD works in continuous (non-terminating) environments
    - MC only works for episodic (terminating) environments
- TD always ground itself more and more correctly progressively

#### Bias/Variance Trade-Off

- In Monte-Carlo
    - Return $G_t$ is an unbiased estimate of $v_{\pi}(S_t)$
    - Here because of discounted rewards of each step the variance of the value function is higher
    - Return depends on many random actions, transitions, rewards

- In TD
    - Using true TD target $R_{t+1} + \gamma v_{\pi}(S_{t+1})$ is also an unbiased estimate of correct value function $v_{\pi}(S_t)$. The true value of the next step is $\gamma v_{\pi}(S_{t+1})$
    - Instead of true value $\gamma v_{\pi}(S_{t+1})$ we use an estimate $\gamma v(S_{t+1})$ which results in biased estimate of $v_{\pi}(S_t)$
    - TD target eventually is much lower variance than the return of MC because we only incur noise from one step
        - $v(S_{t+1})$ is biased as it is not equal to true value $v_{\pi}(S_{t+1})$ but not noisy
        - TD target depends on one random action, transition, reward

- MC has high variance, zero bias
    - Good convergence beacause of zero bias even with function approximation
    - Not very sensitive to initial values as there is not bootstrapping
    - Very simple to understand and use

- TD has low variance, some bias
    - More efficient than MC because of low variance
    - $TD(0)$ converges to true value function $v_{\pi}(s)$ but not always with function approximation
    - More sensitive to initial values because of bootstrapping

#### Batch MC and TD
- MC and TD converge: $V(s) \rightarrow v_{\pi}(s)$ as experience $\rightarrow \inf$
- What about the case of finite experience containing a small set of episodes. Can we find a solution using only that bactch of episodes over and over?
    - e.g. Repeatedly sample episode $k \in [1,K]$ 
    - Apply MC ot $TD(0)$ to episode $k$

- MC converges to solution with minimum mean-squared error
    - Best fit to the observed returns

- $TD(0)$ converges to solution of max likelihood Markov model
    - Solution to the MDP $<S,A,\hat{P},\hat{R},\gamma>$ that best fits the data
    - It first builds the MDP by counting the transitions
    - This MDP can be given to MC to sample from it and work with that samples

- TD exploits Markov Property
    - Usually more efficient in markov environments
- MC does not exploit markov property
    - Usually mode effective in non-markov environments like partially observered, cannot rely on state signals

### Dimensions of Algorithm Space

#### Bootstrapping and Sampling

- Bootsrapping: Update involves an estimated value function instead of using real returns
    - MC does not bootstrap
    - DP and TD bootstraps

- Sampling: Update samples an expectation
    - MC and TD samples the MDP dynamics using a given policy
    - DP does not sample becaue it has to know the complete MDP

### Algorithm between MC and TD

TD($\lambda$) algorithm combines a way to be in between Monte-Carlo and Temporal-Difference learning

#### $n$-Step Prediction

In TD we take a one-step of reality and look at the value function.Instead why not take more than one step, say $n$-steps

- Let TD target look n steps into the future. If n is large it will be similar to MC

- Consider the following $n$-step returns for $n=1,2,...\inf$
    - $n=1$ (TD) $G_t^{(1)} = R_{t+1} + \gamma V(S_{t+1})$
    - $n=2$ $G_t^{(2)} = R_{t+1} + \gamma R_{t+2} + \gamma^2 V(S_{t+2})$
    - .
    - .
    - .
    - .
    - $n=\inf$ (MC) $G_t^{\inf} = R_{t+1} + \gamma R_{t+2}+....+ \gamma^{T-1} R_T$

- So, we can define $n$-step (of real reward) return as
    - $G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3}+...+\gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})$
- Now, we use this new $n$-step return in TD learning
    - $V(S_t) = V(S_t) + \alpha (G_t^{(n)} - V(S_t))$

#### Average $n$-step Returns
- We can average $n$-step returns over different $n$
    - e.g. average the 2-step and 4-step returns
    - $\frac{1}{2}G^{(2)} + \frac{1}{2}G^{(4)}$
- Now the question is how can be efficiently combine information from all the time-steps

#### $\lambda$-return

The algorithm is called TD($\lambda$) which uses the geometrically weighted average of all $n$ going into the future. The constant $\lambda \in [0,1]$ dictates how much we are going to decay the weighting of each successful $n$.

- The $\lambda$-return $G_t^{\lambda}$ combines all n-step returns $G_t^{(n)}$
- Using weight $(1-\lambda) \ \lambda^{n-1}$
    - $G_t^{\lambda} = (1-\lambda) \sum_{n-1}^{\inf} \lambda^{n-1} G_t^{(n)}$
- Forward-view $TD(\lambda)$
    - $V(S_t) \leftarrow V(S_t) + \alpha (G_t^{\lambda} -V(S_t))$
- The reason for using geometric weighting is it memoryless that facilitates an efficiently computable algorithm. Eventually the computational cost of $TD(\lambda)$ is equal to $TD(0)$ using geometric weighting.
- Forward-view is like MC, can only be computed from complete episodes. So, it has the same problems as MC to some extent

#### Backward View TD($\lambda$)
- Forward view provides theory for backward view which is our TD($\lambda$) algorithm
- Backward view provides the mechanism of the algorithm
- Update online, every step, from incomplete sequences

##### Eligibility Traces
- Credit assignment problem: rat example
- Frequency heuristic: assign credit to most frequent states for the value error
- Recency hueristic: assign credit to most recent states for the value error
- Eligibility traces combine both heuristics
    - $E_0(s) = 0$
    - $E_t(s) = \gamma \lambda E_{t-1}(s) + 1 (S_t = s)$

##### Algorithm
- Keep an eligibility trace for every state $s$
- Update the value function $V(s)$ for every state $s$
- In proportion to TD-error $\delta_t$ and eligibility trace $E_t(s)$
    - $\delta_t = R_{t+1} + \gamma V(S_{t+1})-V(S_t)$ i.e. one-step TD error
    - $V(s) \leftarrow V(s) + \alpha \delta_t E_t(s)$ i.e. Update the value function in the direction of TD error according to the credit assignment