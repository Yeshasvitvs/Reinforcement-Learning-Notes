---
tags: Reinforcement Learning, theory, AI
---
# [Reinforcement Learning Lecture 6](https://www.youtube.com/watch?v=UoPei5o4fps&list=PLweqsIcZJac7PfiyYMvYiHfOFPg9Um82B&index=6)
###### David Silver, Deep mind

## Value Function Approximation

Reinforcement learning can be used to solve _large_ scale problems e.g
- Backgammon: $10^{20}$ states
- Computer Go: $10^{170}$ states
- Helicopter (robotics): continuous state space has uncountably numerous states

So, using a lookup table that has a separate value for each state in the state space is not scalable. Hence, the value function are used to have a generalization for the values of the states to represent the environment and to learn the value functions.

How can we scale up the _model-free_ methods for prediction and control?

- So far we have represented value function by a lookup table
    - Every state $s$ has an entry $V(s)$
    - or every state-action pair $s,a$ has an entry $Q(s,a)$
- Problem with large MDPs
    - There are too many states and/or actions to store in memory
    - It is too slow to learn the value of each state individually
- Solution for large MDPs
    - Estimate value function with _function approximation_ 
    - Consider the true value function $v_{\pi}(s)$ as just some function mapping from the state $s$ to $v_{\pi}(s)$
        - $\hat{v}(s,w) \approx v_{\pi}(s)$
        - $\hat{q}(s,a,w) \approx q_{\pi}(s,a)$
    - The way we do it using parametric function approximators with parameter vector $w$, say a vector of weights of a neural network 
    - This provides a compact representation based on the number of parameters rather than the number of states
- Generalize from seen states to unseen states
- Update parameter $w$ using MC or TD learning 

### Types of Value Function Approximation

A neural network is considered to be a canonical function approximator. We can have three different types of architecture for function approximation

- State Value Function Approximation
    - Input query: state $s$
    - Output result: apprroximate state value function $\hat{v}(s,w)$
- In Action Value Function Approximation
    - Input query: state $s$, action $a$
    - Output result: approximate action value function $\hat{q}(s,a,a)$
- Out Action Value Function Approximation
    - Input query: state $s$
    - Output result: approximate action value function for taking all possible actions $\hat{q}(s,a_1,w), \hat{q}(s,a_2,w), ... , \hat{q}(s,a_m,w)$. E.g. atari games

### Which Function Approximator?

There are many function approximators, e.g.
- Linear combinations of features
- Neural networks
- Decision tree
- Nearest neighbor
- Fourier/wavelet bases

We consider _differentiable_ function approximators, where it is relatively easier to adjust the parameter vector after knowing the gradient. we will focus on 
- Linear combinations of features
- Neural network

In reinforcement learning, compared to supervised learning, in practise we end up with non-stationary sequence of value functions we are trying to estimate, non-iid data. So, we require a training method that is suitable for non-stationary, non-iid data. 

### Incremental Methods

#### Gradient Descent

- Let $J(w)$ be a differentiable function (objective function) of parameter vector $w$
- Define the gradient of $J(w)$ to be 
    - $\triangledown_w J(w) = [\frac{\partial J(w)}{\partial w_1}, \frac{\partial J(w)}{\partial w_2},...,\frac{\partial J(w)}{\partial w_n}]^T$
    - This vector tells the direction of steepest descent and we follow it to downhill
- To find a local minimum of $J(w)$
- Adjust the parameter $w$ in the direction of -ve gradient
    - $\Delta w = -\frac{1}{2} \alpha \triangledown_w J(w)$
    - where $\alpha$ is a step-size parameter

##### Value Function Approximation by Stochastic Gradient Descent
- Goal: find the parameter vector $w$ minimizing mean-squared error between approximate value function $\hat{v}(s,w)$ and the true value function $v_{\pi}(s)$
    - $J(w) = E_{\pi}[(v_{\pi}(S)-\hat{v}(S,w))^2]$ consider that we have full knowledge of the correct value function $v_{\pi}(s)$
- Gradient descent finds a local minimum
    - $\Delta w = -\frac{1}{2} \alpha \triangledown_w J(w)$
    - $= \alpha \ E_{\pi}[(v_{\pi}(S)-\hat{v}(S,w)) \triangledown_w \hat{v}(S,w) ]$ by applying the chain rule
    - Expectation is going to be averaging over all the samples we see
    - The way to deal with expectation is using stochastic gradient descent. So, instead of doing full gradient descent i.e.explicitly computing the expectation we will randomly sample a state and consider the difference between the value functions
- Stochastic gradient descent _samples_ the gradient
    - $\Delta w = \alpha (v_{\pi}(S)-\hat{v}(S,w)) \triangledown_w \hat{v}(S,w))$
- Expected update is equal to full gradient update

##### Linear Function Approximators Using Feature Vectors
- Represent a state by a _feature_ vector
    - $x(s) = [x_1(S), x_2(S),...x_n(s)]$
    - Each feature tells something about the state space e.g. distance from landmarks, trends in stockmarket, piece and pawn configurations in chess
    - The whole knowledge of the environment can be encapsulated in the features
- Simplest way to make use of features is to do a linear combination of features based on some weights
- So, a value function by a linear combination of features is given by
    - $$\hat{v}(S,w) = x(S)^T w = \sum_{j=1}^{n} x_j(S) w_j$$
- If the features are extremely sophisticated it will be a perfect representation of the value function
- Objective function is quadratic in parameters $w$
    - $J(w) = E_{\pi}[(v_{\pi}(S) - x(S)^T w)^2]$
- Stochastic gradient descent converges on _global_ optimum
- Update rule is particularly simple
    - $\triangledown_w \hat{v}(S,w) = x(S)$ i.e. gradient is simply features
    - $\Delta w = \alpha \ (v_{\pi}(S) - \hat{v}(S,w) \ x(S))$
- Update = step-size $\times$  prediction error $\times$ feature value

##### Table Lookup Features
- Table lookup is a special case of linear value function approximation
- Using _table lookup_ features
    - $x^{table}(S) = [1(S=s_1), 1(S=s_2),...,1(S=s_n)]^T$
- Parameter vector $w$ gives value of each individual state
    - $\hat{v}(S,w) = [1(S=s_1), 1(S=s_2),...,1(S=s_n)]^T \ . \ [w_1, w_2,...,w_n]^T$

### Incremental Prediction Algorithms
- Till now we assumed we have the true value function $v_{\pi}(s)$ given by some supervisor
- Instead, in RL there is not supervisor but only rewards which will guide our parameter adjustment
- In practive, we substitute a target in place of $v_{\pi}(s)$
    - For MC, the target is the return $G_t$
        - $\Delta w = \alpha (G_t-\hat{v}(S_t,w)) \triangledown_w \hat{v}(S_t,w))$
    - For TD($0$), the target is the TD target, $R_{t+1} + \gamma \hat{v}(S_{t+1},w)$
        - $\Delta w = \alpha (R_{t+1} + \gamma \hat{v}(S_{t+1},w) -\hat{v}(S_t,w)) \triangledown_w \hat{v}(S_t,w))$
    - For TD($\lambda$), the target is the $\lambda$-return $G^{\lambda}_t$
        - $\Delta w = \alpha (G^{\lambda}_t-\hat{v}(S_t,w)) \triangledown_w \hat{v}(S_t,w))$

- It is a bit confusing because, the way we compute rewards is based on the value functions right ?? So, it is a bit like looping here

#### Monte-Carlo with Value Function Approximation
- Return $G_t$ is an unbiased, noisy sample of true values $v_{\pi}(S_t)$
- So, we apply supervised learning to the observed "training data" i.e.
    - $<S_1, G_1>, <S_2,G_2>,...,<S_T,G_T>$
- For example, using lienear MC policy evaluation
    - $\Delta w = \alpha (G_t-\hat{v}(S_t,w)) \triangledown_w \hat{v}(S_t,w)) \Rightarrow \alpha (G_t-\hat{v}(S_t,w)) x(S_t)$
- Monte-carlo evaluation converges to a local optimum
- It converges even when using non-linear value function approximation methods

#### TD Learning with Value Function Approximation
- The TD-target $R_{t+1} + \gamma \hat{v}(S_{t+1},w)$ is a biased sample of true value $v_{\pi}(S_t)$
- This is biased because after one step we invoke our own value function approximator again for the rest of the trajectory
- We can still apply supervised learning to "training data"
    - $<S_1, R_2 + \gamma \hat{v}(S_2,w)>,<S_2, R_3 + \gamma \hat{v}(S_3,w)>,...,<S_{T-1},R_T>$
- E.g. using $TD(0)$
    - $\Delta w = \alpha (R+ \gamma \hat{v})(S^{'},w)-\hat{v}(S,w)) \triangledown_w \hat{v}(S,w)) \Rightarrow \alpha \delta x(S)$
- Even if the samples are biased, linear $TD(0)$ converges close to the global optimum

#### TD($\lambda$) Learning with Value Function Approximation
- The $\lambda$-return $G_t^{\lambda}$ is also a biased sample of true value $v_{\pi}(s)$
- Can apply supervised learning to "training data"
    - $<S_1, G_1^{\lambda}>,<S_2,G_2^{\lambda}>,...,<S_{T-1},G_{T-1}^{\lambda}>$
- Forward view linear TD($\lambda$)
    - $\Delta w = \alpha (G_t^{\lambda}-\hat{v}(S_t,w)) \triangledown_w \hat{v}(S_t,w)) \Rightarrow \alpha (G_t^{\lambda}-\hat{v}(S_t,w)) x(S_t)$
- Backward view linear TD($\lambda$)
    - $\delta_t = R_{t+1} + \gamma \hat{v}(S_{t+1},w) - \hat{v})(S_t,w)$
    - $E_t = \gamma \ \lambda E_{t-1} + x(S_t)$
    - $\Delta w = \alpha \delta_t E_t$
- Forward view and backward view linear $TD(\lambda)$ are equivalent

### Control with Value Function Approximation

Build on the idea of generalized policy iteration

- we start with some parameter vector $w$ that defines some value function
- We act greedily with respect to the value function which gives a new greedy policy
- Policy evaluation: Approximate policy evaluation, $\hat{q}(.,.,w) \approx q_{\pi}$
- Policy improvement: $\epsilon$-greedy policy improvement

#### Action-value Function Approximation

- Approximate the action-value function
    - $\hat{q}(S,A,w) \approx q_{\pi}(S,A)$
- Minimize mean-squared error between approximate action-value functon and true value function
    - $J(w) = E_{\pi} [(q_{\pi}(S,A) - \hat{q}(S,A,w))^2]$
- Use stochastic gradient descent to find a local minimum 
    - $-\frac{1}{2} \triangledown_w J(w) = (q_{\pi}(S,A) - \hat{q}(S,A,w)) \triangledown_w \hat{q}(S,A,w)$
    - $\Delta w = \alpha(q_{\pi}(S,A) - \hat{q}(S,A,q)) \triangledown_w \hat{q}(S,A,w)$

#### Linear Action-Value Function Approximation

- Represent state and action by the feature vector
    - $x(S,A) = [x_1(S,A),...,x_n(S,A)]^T$
- Represent action-value function by linear combination of features
    - $$\hat{q}(S,A,w) = x(S,A)^T w = \sum_{j=1}^{n}x_j(S,A) w_j$$
- Stochastic gradient descent update
    - $\triangledown_w \hat{q}(S,A,w) = x(S,A)$
    - $\Delta w = \alpha(q_{\pi}(S,A) - \hat{q}(S,A,w))x(S,A)$

### Incremental Control Algorithms

- Like prediction, we must substitute a target for $q_{\pi}(S,A)$ i.e. true action-value function
    - For MC, the target is return $G_t$
        - $\Delta w = \alpha (G_t - \hat{q}(S_t,A_t,w)) \triangledown_w \hat{q}(S_t,A_t,w)$
    - For $TD(0)$, the target is the TD target, $R_{t+1} + \gamma Q(S_{t+1},A_{t+1},w)$
        -  $\Delta w = \alpha (R_{t+1} + \gamma \hat{q}(S_{t+1},A_{t+1},w) - \hat{q}(S_t,A_t,w)) \triangledown_w \hat{q}(S_t,A_t,w)$
    - For forward-view $TD(\lambda)$, the target is the action-value $\lambda$-return
        - $\Delta w = \alpha (q_t^{\lambda} - \hat{q}(S_t,A_t,w)) \triangledown_w \hat{q}(S_t,A_t,w)$
    - For backward-view $TD(\lambda)$, equivalent update is
        - $\delta_t = R_{t+1} + \gamma \hat{q}(S_{t+1},A_{t+1},w) - \hat{q}(S_t,A_t,w)$
        - $E_t = \gamma \lambda E_{t-1} + \triangledown_w \hat{q}(S_t,A_t,w)$
        - $\Delta w = \alpha \delta_t E_t$

### Batch Methods

- Incremental methods like gradient descent are simple and appealing but not sample efficient i.e. we experience once and update the value function once and then ignore the experience
- So, batch methods seek to find the best fitting value function to all of the data experience in the batch i.e. _training data_

#### Least Squares Prediction

- Given a value function approximation $\hat{v}(s,w) \approx v_{\pi}(s)$
- with experience $D$ consisting of $<state, value>$ pairs
    - $D = \{ <s_1, v_1^{\pi}>,<s_2,v_2^{\pi}>,...,<s_T,v_T^{\pi}>\}$
- which parameters $w$ give the best fitting value function $\hat{v}(s,w)$?
- Least squares algorithms find parameter vector $w$ by minimizing the sum-squared error between $\hat{v}(s_t,w)$ and true target values $v_t^{\pi}$
    - $$LS(w) = \sum_{t=1}^{T} (v_t^{\pi} - \hat{v}(s_t,w))^2$$
    - $$LS(w) = E_D[(v^{\pi} - \hat{v}(s,w))^2]$$

#### Stochastic Gradient Descent with Experience Replay
- Here we store the whole experience 
    - $D = \{ <s_1, v_1^{\pi}>,<s_2,v_2^{\pi}>,...,<s_T,v_T^{\pi}>\}$

- Repeat for every time-step
    - Sample state, valye from experience
        - $<s,v^{\pi}> \sim D$
- Apply stochastic gradient descent update
    - $\Delta w = \alpha (v^{\pi} - \hat{v}(s,w)) \triangledown_w \hat{v}(s,w)$
- Converges to least squares solution
    - $$w^{\pi} = argmin_w \ LS(w) $$

#### Experience Replay in Deep Q-Networks (DQN)

- DQN uses _experience replay_ and _fixed Q-targets_
    - Take action $a_t$ according to some $\epsilon$-greedy policy
    - Store transition $(s_t,a_t,r_{t+1},s_{t+1})$ in replay memory $D$
    - Sample random mini-batch of transitions $(s,a,r,s^{'})$ from $D$
    - Compute Q-learning targets with respect to old, **fixed** parameters $w^{-}$
    - Optimize MSE between Q-network and Q-learning targets
        - for each iteration
        - $L_i(w_i) = E_{s,a,r,s^{'} \sim D} [(r + \gamma max_{a^{'}} Q(s^{'},a^{'},w_i^{-}) - Q(s,a,w_i))^2]$ 
    - Using variant of stochastic gradient descent
    - The reason to use old paramters to generate targets is not to blowup the network and gives more stable update or else in TD learning everytime we update the $q$ values we update the target values too

##### Example of DQN in Atari

- End-to-end learning of values $Q(s,a)$ from pixels $s$ using convolution neural networks (CNN)
- Input state $s$ is stack of raw pixels from last 4 frames
- Output is $Q(s,a)$ for 18 joystick/button positions
- Reward is change in score for that step

#### Linear Least Squares Prediction

- Experience replay finds least squares solution but it may take huge number of iterations
- If using linear value function approximation $\hat{v}(s,w) x(s)^T w$ we can solve the least squares solution directly
