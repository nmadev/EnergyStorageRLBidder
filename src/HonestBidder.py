import pandas as pd
import numpy as np


class HonestBidder:
    def __init__(
        self,
        initial_soc: float = 0.5,
        capacity: float = 8.0,
        power_max: float = 2.0,
        eff: float = 0.8,
        timestep: float = 5.0 / 60,
    ):
        self.soc = initial_soc
        self.capacity = capacity
        self.power_max = power_max
        self.eff = eff
        self.soc_hist = [initial_soc]
        self.profit_hist = [0.0]
        self.bid_hist = [0.0]
        self.profit_hist = [0.0]
        self.action_hist = [0.0]
        self.timestep = timestep

    def bid(self, lookback: pd.DataFrame) -> tuple[float, tuple[float, float]]:
        """
        Returns the honest bidder's bid given the lookback (current RTP for the bid) and current SoC

        :param lookback: A pandas DataFrame containing all of the historical data
        :return: a tuple containing the bid price and current SoC of the battery
        """
        self.bid_hist.append(lookback["rtp"].iloc[-1])
        return lookback["rtp"].iloc[-1], self.soc

    def step(self, power: float = 0, profit: float = 0) -> None:
        """
        Updates the state of the battery given the power and profit from the previous step.

        :param power: the power request from the grid (- charge, + discharge)
        :param profit: the profit from the previous step
        """
        power_battery = power * self.eff if power < 0 else power / self.eff
        next_soc = self._clamp(
            (self.soc * self.capacity + power_battery * self.timestep) / self.capacity
        )
        self.soc_hist.append(next_soc)
        self.action_hist.append(power)
        self.profit_hist.append(profit)

    def _return_bounds(self) -> tuple[float, float]:
        """
        Returns the maximum charge and discharge power values (as seen by the grid).

        :return: a tuple containing the maximum charge and discharge power values
        """
        charge_bounds = max(
            -(1.0 - self.soc) * self.capacity / self.timestep, -self.power_max
        )
        discharge_bounds = min(self.soc * self.capacity / self.timestep, self.power_max)
        return (-self.power_max, self.power_max)

    def _clamp(value: float, min: float = 0.0, max: float = 1.0) -> float:
        """
        Clamps a value between two bounds.

        :param value: the value to be clamped
        :param min: the minimum value
        :param max: the maximum value
        :return: the clamped value
        """
        return max(min(value, max), min)
