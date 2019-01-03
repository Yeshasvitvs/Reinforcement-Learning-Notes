---
tags: Reinforcement Learning, theory, AI
---
# [Reinforcement Learning Lecture 5](https://www.youtube.com/watch?v=0g4j2k_Ggc4&list=PLweqsIcZJac7PfiyYMvYiHfOFPg9Um82B&index=5)
###### David Silver, Deep mind

## Model Free Control

Some example problems that can be modelled as MDPs
- Elevator
- Parallel Parking
- Ship Steering
- Bioreactor
- Helicoptor
- Aeroplane Logistics
- Robocup Soccer
- Quake
- Portfolio Management
- Robot Walking
- Game of Go

In most of the cases the MDPs are large and often the dynamics are not known correctly. So, you can only sample it and get an experience. Model-free control can be used to solve these problems

### On and Off Policy Learning
- On-policy learning
    - Learning on the job
    - Learn about policy $\pi$ from experience sampled from $\pi$
    - At the same time evaluate the policy based on the actions you take given that policy

- Off-policy learning
    - Look over someone's shoulder
    - Learn about policy $\pi$ from experience sampled from some other agent's policy $\mu$
    - e.g. a robot looking at some other robot or a human behaviour and learn from it by sampling the trajectory done by the other robot or human

We built on top of the policy iteration idea learned before which is based on the full knowledge of the MDP i.e. dynamic programming. So, can we simply use policy iteration with MC evaluation that considers no knowledge of the MDP but instead takes samples from experience of episodes?

Two problems:
- Using only state value function needs MDP knowledge i.e. $p^a_{ss^{'}}$
- When you act greedily you are not exploring any possible potential states available in the environment

- Greedy policy improvement over $V(s)$ requires model of MDP
    - $\pi^{'}(s) = argmax_{a \in A} R_s^a + P^a_{ss^{'}}V(s^{'})$

- Greedy policy improvement over $Q(s,a)$ allows to be model-free
    - $\pi^{'}(s) = argmax_{a \in A} Q(s,a)$
    - reason is to maximizing over $q$ values, so we can pick an action that maximizes $q$ values

#### Generalized Policy Iteration with Action-Value Functuin
- Policy Evaluation: Monte-Carlo policy evaluation, $Q = q_{\pi}$
- Policy Improvement: Greedy policy improvement will not work in this case as we will be stuck to some local minima that will not give us the best policy as there is no exploration. So we need a new greedy exploration

##### $\epsilon$-Greedy Exploration
- Simplest idea for ensuring continual exploration
- All $m$ actions are tried with non-zero probability
- with probability $1-\epsilon$ choose the greedy action
- with probability $\epsilon$ choose an action at random
    - $\pi(a|s) = \epsilon/m + 1 - \epsilon \ is \ a^* = argmax_{a \in A} \ Q(s,a); \ \epsilon/m \ \ otherwise$
- This guarantees policy improvement

##### $\epsilon$-Policy Improvement$
- Theorem:
    - For any $\epsilon$-greedy policy $\pi$, the $\epsilon$-greedy policy $\pi^{'}$ with respect to $q_{\pi}$ is an improvement, $v_{\pi^{'}}(s) \ge v_{\pi}(s)$
    - The proof is similar to the one shown in dynamic programming policy iteraction

##### Monte-Carlo Policy Iteration
Now the policy iteartion process for model free will become
- Policy Evaluation: Monte-Carlo policy evaluation, $Q = q_{\pi}$
- Policy Improvement: $\epsilon$-greedy policy improvement
    - The policy is always stochastic now because of the use of probability based on $\epsilon$ which shows that we explore the environment at some rate.
    - The rate of exploration can be a bit slower which gives no strong guarantee on the time to convergence to optimal policy

##### Monte-Carlo Control
- To improve the time taken for convergence, we can revisit the idea of not necessarily to fully evaluate a policy (check dynamic programming) using many iterations
- The idea is to use the latest estimate to improve the policy
- We look at only one episode instead of doing policy evaluation over many episodes

##### GLIE

How can we guarantee that the best possible policy will be found ${\pi}_{*}$ including a possibility of exploration. How do we balance these two ideas is using _Greedy in the Limit with Infinite Exploration (GLIE)_

In GLIE we come up with any schedule for exploration such that two conditions are met,
- First condition is you continue to explore everything 
    - All state-action pairs are explored infinitely many times,
    - $$\lim_{k\to\infty} N_k(s,a) = \infty$$
- Secondly, we want to make sure the policy eventually becomes greedy because it has to satisfy bellman optimality equation which has some maximum
    - The policy converges on a greedy policy,
    - $$\lim_{k \to\infty}{\pi}_k(a|s) = 1(a=argmax_{a^{'} \in A} Q_k(s,a^{'}))$$

- e.g. $\epsilon$-greedy is GLIE if $\epsilon$ reduces to zero at $\epsilon_k = \frac{1}{k}$

##### GLIE Monte-Carlo Control Algorithm
- We start by sampling episodes till we get to some terminal state $S_T$ using a given policy $\pi$
    - So the trajectory contains $\{S_1,A_1,R_2,...,S_t\} \sim \pi$
- For each of those states and actions in the episode, we update the action value (this is policy evaluation)
    - $N(S_t,A_t) \leftarrow N(S_t,A_t) + 1$ i.e. incremental counting of how many times we saw a state-action pair
    - $Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \frac{1}{N(S_t,A_t)} (G_t - Q(S_t, A_t))$ i.e. incremental update to the mean
- It is interesting that we are not taking a mean over IID quantities, the policy actually changing over time when we are improving it and we are taking returns from better and better policies
- Improve policy based on new action-value function, the GLIE property ensures that the means we are collecting over time converge to mean return for optimal policy
    - $\epsilon \leftarrow \frac{1}{k}$
    - $\pi \leftarrow \epsilon-greedy(Q)$
- We iterate over multiple episodes like this
- Theorem
    - GLIE Monte-Carlo control converges to the optimal action-value function, $Q(s,a) \rightarrow q_{*}(s,a)$

#### MC vs. TD Control
- TD learning has several advantages over MC
    - Lower variance
    - Online
    - Incomplete sequences
- Natural idea: use TD insteal of MC in our control loop
    - Apply TD to $Q(s,a)$
    - Use $\epsilon$-greedy policy improvement
    - Update every time-step

##### Updating Action-Value Functions with Sarsa
- Idea: we start with a state-action pair $(S,A)$, we randomly sample from the environment and see a reward $R$, we will end up in a new state $S^{'}$ and we sample using our policy and end up at the next state $A$. Hence the name SARSA
- $Q(S,A) \leftarrow Q(S,A) + \alpha (R + \gamma Q(S^{'},A^{'}) - Q(S,A))$

##### On-Policy Control with Sarsa 

For each time-step
- Policy evaluation, Sarsa, $Q \approx q_{\pi}$
- Policy improvement. $\epsilon$-greedy policy improvement

###### Algorithm

- Initialize $Q(s,a), \forall s \in S, a \in A(s)$, _arbitarily_, and $Q(terminal state,.) = 0$
- Repeat (for each episode):
    - Initialize $S$
    - Choose $A$ from $S$ using policy derived from $Q$ (e.g., $\epsilon$-greedy)
    - Repeat (for each step of episode):
        - Take action $A$, observe $R$, $S^{'}$
        - Choose new action $A^{'}$ from $S^{'}$ using policy derived from $Q$ (e.g. $\epsilon$=greedy)
        - $Q(S,A) \leftarrow Q(S,A) + \alpha (R + \gamma Q(S^{'},A^{'}) - Q(S,A))$
        - $S \leftarrow S^{'}; A \leftarrow A^{'}$
    - Until $S$ is a terminal state

##### Convergence of Sarsa
Theorem
- Sarsa converges to the optimal action-value function, $Q(s,a) \rightarrow q_{\pi}(s,a)$, under the following conditions
    - GLIE sequence of policies ${\pi}_t(a|s)$
    - Robbins-Monro sequence of step-size $\alpha_t$
        - $$\sum_{t = 1}^{\infty} \alpha_t = \infty$$
        - $$\sum_{t = 1}^{\infty} \alpha^2_t < \infty$$

##### $n$-step Sarsa    
- Define the $n$-step $Q$-return
    - $q^{(n)}_t = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{n-1} R_{t+n} + \gamma^n Q(S_{t+n})$

- $n$-step Sarsa updates $Q(s,a)$ towards the $n$-step $Q$-return
    - $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha (q^{(n)}_t - Q(S_t, A_t))$

##### Forward View Sarsa($\lambda$)

- The $q^n$ return combines all $n$-step $Q$-returns $q^{(n)}_t$
- Using weight $(q- \lambda) \lambda^{n-1}$
    - $$q^{\lambda}_t = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} q^{(n)}_t$$
- Forward view Sarsa($\lambda$)
    - $Q(S_t,A_t) \leftarrow Q(S_t, A_t) + \alpha \ (q^{\lambda}_t - Q(S_t, A_t))$

##### Backward View Sarsa($\lambda$)
- Just like TD($\lambda$), we use eligibility traces in an online algorithm
- But Sarsa($\lambda$) has one eligibility trace for each state-action pair
    - $E_0(s,a) = 0$
    - $E_t(s,a) = \gamma \lambda E_{t-1}(s,a) + 1(S_t = s, A_t = a)$
- $Q(s,a)$ is updated for every state $s$ and action $a$ pair 
- In proportion to TD-error $\delta_t$ and eligibility trace $E_t(s,a)$
    - $\delta_t = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)$
    - $Q(s,a) \leftarrow Q(s,a) + \alpha \delta_t E_t(s,a)$

### Off-Policy Learning
- On policy learning is about the policy I am following
- Off policy learning involves evaluating target policy $\pi(a|s)$ to compute $v_{pi}(s)$ or $q_{\pi}(s,a)$ while following a behaviour policy $\mu(a|s)$ 
    - $\{S_1, A_1, R_2,....S_T\} \approx \mu$
- Examples
    - Learn from observing humans or other agents
    - Re-use experience generated from old policies $\pi_1, \pi_2,...,\pi_{t-1}$
    - Learn about optimal policy while following _exploratory_ policy
    - Learn about _multiple_ policies while following _one_ policy

- Two mechanisms for dealing with off-policy learning are:
    -   Importance Sampling
    -   Q-Learning

#### Importance Sampling
This involved estimating the expectation of a different distribution

- $E_{X \sim P }[f(X)] = \sum P(X) \ f(X)$ (sum expectation of future reward is sum over some probabilities times how much reward we get)
-   $= \sum Q(X) \frac{P(X)}{Q(X)} \ f(X)$ (multiplying with some other distribution)
-   $= E_{X \sim Q} \frac{P(X)}{Q(X)} \ f(X)$ (this is now sum expectation of the new distribution)
-   This gives the correction between the changes in the distribution

#### Importance Sampling for Off-Policy Monte-Carlo
- Use returns generated from $\mu$ to evaluatre $\pi$
- Weight return $G_t$ according to similarity between policies
- Multiply importance sampling corrections along the whole episode i.e. trajectory
    - $G_t^{\frac{\pi}{\mu}} = \frac{\pi(A_t|S_t)\pi(A_{t+1}|S_{t+1})...\pi(A_T,S_T)}{\mu(A_t|S_t)\mu(A_{t+1}|S_{t+1})...\mu(A_T,S_T)} G_t$
- Update value towards _corrected_ return
    - $V(S_t) \leftarrow V(S_t) + \alpha \ (G_t^{\frac{\pi}{\mu}} - V(S_t))$
- Importance sampling can dramatically increase variance and hence is not usable in practise 
- Cannot use if $\mu$ is zero when $\pi$ is non-zero
- So, off-policy MC is not a good idea

#### Importance Sampling for Off-Policy TD
- Use TD targets generated from $\mu$ to evaluate $\pi$
- Weight TD target $R + \gamma V(S^{'})$ of one step by importance sampling 
- Bootstrapping is imperative in TD
- Only need a single importance sampling correction
    - $V(S_t) \leftarrow V(S_t) + \alpha \ (\frac{\pi(A_t|S_t)}{\mu(A_t|S_t)}(R_{t+1} + \gamma V(S_{t+1}) - V(S_t))$
- Much lower variable than MC as policies only need to be similar over a single step

### Q-Learning

This is the idea that works best with off-policy learning
- We now consider off-policy learning of action-values $Q(s,a)$
- No importance sampling is required
- Idea: 
    - Next action is chosen using the behaviour policy $A_{t+1} \approx \mu(s|S_t)$
    - But we consider alternative successor action $A^{'} \approx \pi(.|S_t)$ that we might take if we are taking target policy
    - Update $Q(S_t, A_t)$ towards value of alternative action
        - $Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha \ (R_{t+1} + \gamma Q(S_{t+1}, A^{'}) - Q(S_t,A_t))$
        - So, the boot strapping is with the action-value function related to the alternative action

#### Off-Policy Control with Q-Learning
- A special case where the target policy is a greedy policy i.e. we tried to learn some greedy behaviour while we are following some exploratory behaviour
- We now allow both behaviour and target policies to improve
- The target policy $\pi$ is greedy w.r.t $Q(s,a)$
    - $\pi(S_{t+1}) = argmax_{a^{'}} \ Q(S_{t+1}, a^{'})$
- The behaviour policy $\mu$ is e.g. $\epsilon$-greedy w.r.t $Q(s,a)$
- The Q-learning target then simplifies
    - $R_{t+1} + \gamma \ Q(S_{t+1}, A^{'})$
    - $R_{t+1} + \gamma \ Q(S_{t+1}, argmax_{a^{'}} \ Q(S_{t+1}, a^{'}))$
    - $R_{t+1} + max_{a^{'}} \ \gamma \ Q(S_{t+1}, a^{'})$

##### Q-Learning Control Algorithm SARSAMAX

$Q(S,A) \leftarrow Q(S,A) + \alpha \ (R + \gamma \ max_{a^{'}} Q(S^{'}, a^{'}) - Q(S,A))$

- This is similar to SARSA and updating the Q values in the direction of the best possible Q value you could have by next step 

Theorem
- Q-learning control converges to the optimal action-value function, $Q(s,a) \rightarrow q_{*}(s,a)$


### Relationship Between DP and TD

|  | Full Backup (DP) | Sample Backup (TD) |
| :--------: | :--------: | :--------: |
| Bellman Expectation Equation for $v_{\pi}(s)$    | Iterative Policy Evaluation     | TD Learning     |
| Bellman Expectation Equation for $q_{\pi}(s,a)$    | Q-Policy Iteration     | Sarsa     |
| Bellman Optimality Equation for $q_{*}(s)$    | Q-Value Iteration     | Q-Learning     |


