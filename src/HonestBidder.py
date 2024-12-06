import pandas as pd
import numpy as np


class HonestBidder:
    def __init__(
        self,
        initial_soc: float = 0.5,
        capacity: float = 8.0,
        power_max: float = 2.0,
        eff: float = 0.8,
        granularity: float = 5.0 / 60,
        noise: tuple[float, float] = (0.0, 0.0),
    ):
        self.initial_soc = initial_soc
        self.soc = self.initial_soc
        self.capacity = capacity
        self.power_max = power_max
        self.eff = eff
        self.soc_hist = [self.initial_soc]
        self.profit_hist = [0.0]
        self.bid_hist = [0.0]
        self.action_hist = [0.0]
        self.granularity = granularity
        self.noise = noise

    def reset(self) -> None:
        """
        Resets the state of the battery
        """
        self.soc = self.initial_soc
        self.soc_hist = [self.initial_soc]
        self.profit_hist = [0.0]
        self.bid_hist = [0.0]
        self.action_hist = [0.0]

    def bid(self, lookback: pd.DataFrame) -> tuple[float, tuple[float, float]]:
        """
        Returns the honest bidder's bid given the lookback (current RTP for the bid) and current SoC

        :param lookback: A pandas DataFrame containing all of the historical data
        :return: a tuple containing the bid price and current SoC of the battery
        """
        noisy_bid = lookback.rtp.iloc[-1] + np.random.normal(
            loc=self.noise[0], scale=self.noise[1]
        )
        self.bid_hist.append(noisy_bid)
        power_bounds = self._return_bounds()
        return noisy_bid, power_bounds

    def step(self, power: float = 0, profit: float = 0) -> None:
        """
        Updates the state of the battery given the power and profit from the previous step.

        :param power: the power request from the grid (- charge, + discharge)
        :param profit: the profit from the previous step
        """
        power_battery = power * self.eff if power < 0 else power / self.eff
        next_soc = (
            self.soc * self.capacity - power_battery * self.granularity
        ) / self.capacity
        # TODO: Check this... it shouldn't be necessary to clamp the SoC
        next_soc = self._clamp(value=next_soc)
        self.soc_hist.append(next_soc)
        self.soc = next_soc
        self.action_hist.append(power)
        self.profit_hist.append(profit)

    def _return_bounds(self) -> tuple[float, float]:
        """
        Returns the maximum charge and discharge power values (as seen by the grid).

        :return: a tuple containing the maximum charge and discharge power values
        """
        charge_bounds = max(
            -(1.0 - self.soc) * self.capacity / self.granularity, -self.power_max
        )
        discharge_bounds = min(
            self.soc * self.capacity / self.granularity, self.power_max
        )
        return (charge_bounds, discharge_bounds)

    def _clamp(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """
        Clamps a value between two bounds.

        :param value: the value to be clamped
        :param min_val: the minimum value
        :param max_val: the maximum value
        :return: the clamped value
        """
        return max(min(value, max_val), min_val)
