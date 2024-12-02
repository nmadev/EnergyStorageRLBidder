import numpy as np
import pandas as pd


class BidSimulator:
    """
    This class contains an object to simulate an energy market bidder.
    """

    def __init__(
        self,
        data: pd.DataFrame = None,
        lookback_periods: int = 0,
        # TODO: Implement a start parameter that functions the same as the end parameter
        end: int | pd.Timestamp = None,
        eff: float = 1.0,
        discharge_cost: float = 10.0,
        initial_soc: float = 1.0,
        capacity: float = 10.0,
        power_max: float = 2.0,
        resting_draw: float = 0.01,
        timestep: float = 5.0 / 60,
        params: dict = None,
        bid_acceptance_probability: float = 0.5,
        price_volatility: float = 20.0,
    ):
        """
        Initializes a BidSimulator object.

        :param data: Pandas DataFrame containing columns real-time (rtp), timestamp (ts), and any other relevant data
        :param lookback_periods: number of periods contained in lookback window (state encoding)
        :param end: last timestep or number of periods allowed in simulation
        :param eff: one way efficiency for charging/discharging the battery (default 1.0)
        :param discharge_cost: cost associated with discharging ($/MW)
        :param initial_soc: initial state-of-charge [0, 1] (unitless)
        :param capacity: capacity of the battery (MWh)
        :param power_max: maximum charge/discharge power (before efficiency corrections) of the battery (MW)
        :param resting_draw: resting draw (MW)
        :param timestep: number of hours in a single timestep (hours^-1)
        :param params: dictionary of additional parameter used by the bidding model
        """
        self.data = data.sort_values(by="ts").reset_index()
        self.ts_idx = lookback_periods
        self.ts = data.ts.iloc[self.ts_idx]
        self.end = len(data) if end == None else end
        self.eff = eff
        self.discharge_cost = discharge_cost
        self.capacity = capacity
        self.power_max = power_max
        self.resting_draw = resting_draw
        self.timestep = timestep
        self.params = params
        self.soc_hist = [initial_soc]
        self.profit_hist = [0]
        self.action_hist = [np.nan]
        self.price_hist = [np.nan]
        self.done = False
        # initial parameters
        self.init_params = {
            "lookback_periods": lookback_periods,
            "initial_soc": initial_soc,
        }
        self.bid_acceptance_probability = bid_acceptance_probability
        self.price_volatility = price_volatility

    def __str__(self):
        self_str = f"""Bid Summary
\tSOC: {self.soc_hist[-1]}
\tProfit: {self.profit_hist[-1]}
\tTimestamp: {self.ts_idx}/{len(self.data)}
\tCapacity: {self.capacity}
\tMaximum Power: {self.power_max}
\tEfficiency: {self.eff}
\tDischarge Cost: {self.discharge_cost}
\tResting Draw: {self.resting_draw}"""
        return self_str

    def reset_simulation(
        self,
    ):
        self.done = False
        self.ts_idx = self.init_params["lookback_periods"]
        self.ts = self.data.iloc[self.ts_idx]
        self.soc_hist = [self.init_params["initial_soc"]]
        self.profit_hist = [0]
        self.action_hist = [np.nan]
        self.price_hist = [np.nan]

    def step(self, bid: float = 0.0):
        """
        Steps the simulation by one time period

        :return: next state of charge, profit from the previous step (can be negative), max discharge/charge values and terminal state status
        """
        if not self.done:
            real_time_price = self.data.rtp.iloc[self.ts_idx]
            price_noise = np.random.normal(0, self.price_volatility)
            noisy_price = max(0, real_time_price + price_noise)

            next_soc, profit, power_bounds = self._bid(noisy_price, bid)

            self.ts_idx += 1
            self.ts = self.data.iloc[self.ts_idx].ts

            done = False
            curr_ts = self.ts_idx if type(self.end) == int else self.ts
            if curr_ts > self.end:
                done = True
                return -1, -1, [0, 0], done, self.data.copy()
            self.done = done
            return (
                next_soc,
                profit,
                power_bounds,
                done,
                self.data[self.data.index <= self.ts_idx].copy(),
            )
        return -1, -1, [0, 0], self.done, self.data.copy()

    def _bid(self, price: float = None, power: float = None):
        """
        Updates simulation parameters by charging/discharging at a certain capacity from a given bid.

        :param price: current price of electricity ($/MW)
        :param power: power into battery (+ charge, - discharge) observed by grid
        :return: next state of charge, profit from the previous step (can be negative), and max discharge/charge values
        """
        assert np.abs(power) <= self.power_max
        if np.random.random() < self.bid_acceptance_probability:
            corrected_power = power / self.eff if power < 0 else power * self.eff
            next_soc = max(
                self.soc_hist[-1]
                + (corrected_power - self.resting_draw) * self.timestep / self.capacity,
                0,
            )
            profit = -power * price
            power_bounds = [
                max(-next_soc * self.capacity / self.timestep, -self.power_max),
                min((1.0 - next_soc) * self.capacity / self.timestep, self.power_max),
            ]
            assert next_soc <= 1.0
            assert next_soc >= 0.0
        else:
            next_soc = self.soc_hist[-1]
            profit = 0
            power_bounds = self.get_action()

        # update battery attributes
        self.action_hist.append(power)
        self.soc_hist.append(next_soc)
        self.profit_hist.append(profit)

        # return parameters for the next step
        return next_soc, profit, power_bounds

    def get_state(self):
        """
        Returns the current state of the bidder.

        :return: current soc, current total profit, done, and lookback dataframe
        """
        return (
            self.soc_hist[-1],
            sum(self.profit_hist),
            self.done,
            self.data[self.data.index <= self.ts_idx],
        )

    def get_action(self):
        """
        Returns the allowable actions (power bounds) of the bidder

        :return: array representing the maximum discharge and charge actions
        """
        return [
            max(-self.soc_hist[-1] * self.capacity / self.timestep, -self.power_max),
            min(
                (1.0 - self.soc_hist[-1]) * self.capacity / self.timestep,
                self.power_max,
            ),
        ]

    def get_summary(self):
        """
        Build a summary dictionary of run statistics and returns as a dictionary.

        :return: dictionary of summary statistics
        """
        summary = {
            "soc_history": self.soc_hist,
            "profit_history": self.profit_hist,
            "power_history": self.action_hist,
            "price_history": self.price_hist,
            "battery_capacity": self.capacity,
            "max_power": self.power_max,
            "efficiency": self.eff,
            "discharge_cost": self.discharge_cost,
        }
        return summary
