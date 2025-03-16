# Markov Model (a.k.a. Markov Chain)

A Markov model is a stochastic model used to represent a sequence of possible events where the probability of each event depends only on the state attained in the previous event. This property is known as the Markov property, which states that the future state depends only on the current state and not on the sequence of events that preceded it.

Mathematically, for a discrete-time Markov chain, the Markov property can be expressed as:

$$
P(Y_{t+1} = y \mid Y_t = y_t, Y_{t-1} = y_{t-1}, \dots, Y_0 = y_0) = P(Y_{t+1} = y \mid Y_t = y_t)
$$

where:
- $ Y_t $ denotes the state of the system at time $ t $.
- The conditional probability indicates that the next state depends only on the current state, not on the entire sequence of past states.

Using the rules of independence, we can calculate the joint probability of the sequence as: 

$$
P(y_1,...,y_T) = P(y_1)P(y_2|y_1)...P(y_{T}|y_{T-1}) = P(y_1) \prod_{t=2}^T P(y_t|y_{t-1})
$$

Markov chains assume that the conditional probability $P(y_t|y_{t-1})$ does not vary with time. Therefore, we can fully specify a Markov chain using three parameters:

- Number of States $(M)$: We generally assume that the observation can take one of $M$ states.

- Transition Matrix $(A)$: The transition matrix stores the probability of transition between the state $i$ to state $j$. Thus, the transition matrix can be represented as a $M \times M$ matrix where the entry $A_{ij}$ is given by $A_{ij} = P(y_t = j \mid y_{t-1} = i)$ where $i, j \in \{1, 2, \ldots, M\}$.

- Prior Probability $(\pi)$: The probability of starting from one of the available states, denoted by $\pi_i = P(y_1 = i)$ where $i \in \{1, 2, \ldots, M\}$.

# Discrete Hidden Markov Model (HMM)

In a discrete Hidden Markov Model (HMM), discrete observations $y_t$ are generated based on discrete hidden states $x_t$. 

Formally, an HMM is defined by the following components:

1. **Hidden States**: A finite set of $ K $ hidden states $ S = \{x_1, x_2, \ldots, x_K\} $.

2. **Transition Matrix $ (A) $**: An $ K \times K $ matrix where each entry $ A_{ij} = P(x_j \mid x_i) $ represents the probability of transitioning from state $ x_i $ to state $ x_j $.

3. **Emission Matrix $ (B) $**: An $ K \times M $ matrix where each entry $ B_{ij} = P(y_j \mid x_i) $ represents the probability of emitting observation $ y_j $ from state $ x_i $, and $ M $ is the number of possible observations.

4. **Initial State Distribution $ (\pi) $**: A vector $ \pi $ where each entry $ \pi_i = P(x_i) $ represents the probability of the system starting in state $ x_i $.

The joint probability of a sequence of hidden states $X = \{x_1, x_2, \ldots, x_T\} $ and observations $ Y = \{y_1, y_2, \ldots, y_T\} $ is given by:

$$
P(X,Y) = \pi_{x_1} \prod_{t=2}^T A_{x_{t-1}x_t} \prod_{t=1}^T B_{x_t y_t}
$$

The key properties of HMMs include:

- **Markov Property**: The probability of transitioning to a future state depends only on the current state and not on the sequence of events that preceded it.

$$
P(x_{t+1} \mid x_t, x_{t-1}, \dots, x_1) = P(x_{t+1} \mid x_t)
$$

- **Emission Independence**: Each observation $ y_t $ depends only on the corresponding hidden state $ x_t $ and is conditionally independent of other observations given the hidden states.

# SSMs

A state-space model (SSM) is a partially observed Markov model, in which the hidden state, $ z_t $, evolves over time according to a Markov process, and each hidden state generates some observations $ y_t $ at each time step. The main goal is to infer the hidden states given the observations. However, we may also be interested in using the model to predict future observations (e.g., for time-series forecasting).

An SSM can be represented as a stochastic discrete time nonlinear dynamical system of the form

$$
z_t = f(z_{t-1}, u_t, q_t)
$$

$$
y_t = h(z_t, u_t, y_{1:t-1}, r_t)
$$

where $ z_t \in \mathbb{R}^{N_z} $ are the hidden states, $ u_t \in \mathbb{R}^{N_u} $ are optional observed inputs, $ y_t \in \mathbb{R}^{N_y} $ are observed outputs, $ f $ is the transition function, $ q_t $ is the process noise, $ h $ is the observation function, and $ r_t $ is the observation noise.

Rather than writing this as a deterministic function of random noise, we can represent it as a probabilistic model as follows:

$$
p(z_t|z_{t-1}, u_t) = p(z_t|f(z_{t-1}, u_t))
$$

$$
p(y_t|z_t, u_t, y_{1:t-1}) = p(y_t|h(z_t, u_t, y_{1:t-1}))
$$

where $ p(z_t|z_{t-1}, u_t) $ is the transition model, and $ p(y_t|z_t, u_t, y_{1:t-1}) $ is the observation model.

Unrolling over time, we get the following joint distribution:

$$
p(y_{1:T}, z_{1:T} | u_{1:T}) = \left[ p(z_1|u_1) \prod_{t=2}^{T} p(z_t|z_{t-1}, u_t) \right] \left[ \prod_{t=1}^{T} p(y_t|z_t, u_t, y_{1:t-1}) \right]
$$

If we assume the current observation $ y_t $ only depends on the current hidden state, $ z_t $, and the previous observation, $ y_{t-1} $, we get the graphical model in Figure 29.1(a). (This is called an auto-regressive state-space model.) However, by using a sufficient expressive hidden state $ z_t $, we can implicitly represent all the past observations, $ y_{1:t-1} $. Thus it is more common to assume that the observations are conditionally independent of each other (rather than having Markovian dependencies) given the hidden state. In this case the joint simplifies to

$$
p(y_{1:T}, z_{1:T} | u_{1:T}) = \left[ p(z_1|u_1) \prod_{t=2}^{T} p(z_t|z_{t-1}, u_t) \right] \left[ \prod_{t=1}^{T} p(y_t|z_t) \right]
$$

Sometimes there are no external inputs, so the model further simplifies to the following unconditional generative model:

$$
p(y_{1:T}, z_{1:T}) = \left[ p(z_1) \prod_{t=2}^{T} p(z_t|z_{t-1}) \right] \left[ \prod_{t=1}^{T} p(y_t|z_t) \right]
$$
