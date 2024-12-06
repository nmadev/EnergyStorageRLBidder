import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd


class DQNAgent:
    def __init__(self,
                 lr,
                 prob_clear,
                 attitude,
                 data,
                 initial_soc: float = 0.5,
                 capacity: float = 8.0,
                 power_max: float = 2.0,
                 eff: float = 0.8,
                 granularity: float = 5.0 / 60):
        """
        DQN Agent initialization.
        :param lr: Learning rate for optimizer
        :param prob_clear: Function to calculate bid acceptance probability
        :param attitude: attitude of the agent
        :param data: Loaded data containing fields rtp, dap, timestamp
        """
        self.obssize = 4  # observation size (RTP, DAP, SOC, Timestamp)
        self.action_space = np.arange(0.0, 2.1, 0.1)
        self.actsize = len(self.action_space)

        # Neural Network Model Definition
        self.model = nn.Sequential(
            nn.Linear(self.obssize, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.actsize)
        )

        # Optimizer Definition
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.data = data
        self.prob_clear = prob_clear
        self.attitude = attitude
        self.capacity = capacity
        self.power_max = power_max
        self.eff = eff
        self.soc_hist = [initial_soc]
        self.profit_hist = [0.0]
        self.bid_hist = [0.0]
        self.action_hist = [0.0]
        self.power_hist = []
        self.granularity = granularity

    def step(self, power, profit, lookback):
        """
        Step function to simulate the environment dynamics.

        :return: next_state
        """
        soc = self.soc_hist[-1]
        timestamp = lookback['ts'].iloc[-1]

        # Find the next timestamp in the dataset
        next_index = (self.data['ts'] > timestamp).idxmin()

        next_rtp = self.data.loc[next_index, 'rtp']
        next_dap = self.data.loc[next_index, 'dap']
        next_timestamp = self.data.loc[next_index, 'ts'].timestamp()

        power_battery = power * self.eff if power < 0 else power / self.eff
        next_soc = self._clamp(
            (soc * self.capacity - power_battery * self.granularity) / self.capacity
        )

        self.soc_hist.append(next_soc)
        self.action_hist.append(power)
        self.profit_hist.append(profit)

        # Construct next state
        next_state = np.array([next_rtp, next_dap, next_soc, float(next_timestamp)])

        return next_state

    def bid(self, lookback: pd.DataFrame):
        # Convert the necessary columns to numeric values
        rtp = lookback["rtp"].iloc[-1]
        dap = lookback["dap"].iloc[-1]
        ts = lookback["ts"].iloc[-1].timestamp()  # Convert to Unix timestamp

        # Construct the state with numeric values only
        state = np.array([float(rtp), float(dap), float(self.soc_hist[-1]), float(ts)])

        action_index = self.compute_argmaxQ(state)
        action = self.action_space[action_index]
        power_bounds = self._return_bounds()
        return rtp * action, power_bounds, action_index

    def train(self, buffer, gamma, initialsize, batchsize, tau, episodes):
        """
        Train the DQN Agent using the loaded data.

        :param buffer: Replay buffer for experience storage
        :param gamma: Discount factor
        :param initialsize: Minimum number of experiences before training starts
        :param batchsize: Batch size for training
        :param tau: Target network update frequency
        :param episodes: Number of episodes to train for
        """
        rrecord = []
        totalstep = 0

        for episode in range(episodes):
            # Initialize state from data
            initial_index = 0  # Start from the beginning of the dataset
            state = np.array([self.data.loc[initial_index, 'rtp'],
                              self.data.loc[initial_index, 'dap'],
                              0.5,  # Initial SOC
                              float(self.data.loc[initial_index, 'ts'].timestamp())])
            rsum = 0
            for i in range(20, len(self.data)):
                # Step Function
                rtp, dap, soc, ts = state

                # Epsilon-Greedy Action Selection
                if np.random.rand() < max(0.01, 0.9999 ** episode):
                    action_index = np.random.choice(len(self.action_space))
                    action = self.action_space[action_index]
                    bid = rtp * action
                else:
                    bid, power_bounds, action_index = self.bid(self.data.iloc[:i])

                prob_cleared = self.prob_clear(rtp, bid, self.attitude, soc)

                power = 0
                if prob_cleared == 1:
                    power = self._return_bounds()[1]
                elif prob_cleared == -1:
                    power = self._return_bounds()[0]

                self.power_hist.append(power)

                # Reward Calculation
                reward = power * rtp * self.granularity

                next_state = self.step(power, reward, self.data.iloc[:i])

                # Append Experience to Buffer
                buffer.append((state, action_index, reward, next_state))

                # Train Agent
                if totalstep > initialsize and len(buffer.buffer) >= batchsize:
                    minibatch = buffer.sample(batchsize)
                    states, actions, rewards, next_states = zip(*minibatch)

                    max_next_q_values = self.compute_maxQvalues(next_states)
                    targets = [r + gamma * max_next_q for r, max_next_q in
                               zip(rewards, max_next_q_values)]

                    self.train_step(states, actions, targets)

                # Update Target Network
                if totalstep % tau == 0:
                    self.model.load_state_dict(self.model.state_dict())

                totalstep += 1
                rsum += reward
                state = next_state

            rrecord.append(rsum)
            if episode % 10 == 0:
                print(f'Episode {episode}, Average Return: {np.mean(rrecord[-10:])}')

    def compute_argmaxQ(self, state):
        """
        Compute the action that maximizes Q for a given state.
        """
        state = torch.FloatTensor(state)
        Qvalue = self.model(state).detach().numpy()
        return np.argmax(Qvalue.flatten())

    def compute_maxQvalues(self, states):
        """
        Compute max Q-values for a batch of states.
        """
        states = torch.FloatTensor(states)
        Qvalues = self.model(states).detach().numpy()
        return np.max(Qvalues, axis=1)

    def train_step(self, states, actions, targets):
        """
        Train the agent on a given batch of states, actions, and targets.

        states: numpy array as input to compute loss (s)
        actions: numpy array as input to compute loss (a)
        targets: numpy array as input to compute loss (Q targets)
        """
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).view(-1, 1)
        targets = torch.FloatTensor(targets)

        q_preds_selected = self.model(states).gather(1, actions).squeeze()

        loss = torch.mean((q_preds_selected - targets) ** 2)

        # Backward Pass
        self.optimizer.zero_grad()
        loss.backward()

        # Update Weights
        self.optimizer.step()
        return loss.detach().cpu().data.numpy()

    def _return_bounds(self) -> tuple[float, float]:
        """
        Returns the maximum charge and discharge power values (as seen by the grid).

        :return: a tuple containing the maximum charge and discharge power values
        """
        soc = self.soc_hist[-1]
        charge_bounds = max(
            -(1.0 - soc) * self.capacity / self.granularity, -self.power_max
        )
        discharge_bounds = min(soc * self.capacity / self.granularity, self.power_max)
        return (charge_bounds, discharge_bounds)

    def _clamp(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """
        Clamps a value between two bounds.

        :param value: the value to be clamped
        :param min: the minimum value
        :param max: the maximum value
        :return: the clamped value
        """
        return max(min(value, max_val), min_val)
