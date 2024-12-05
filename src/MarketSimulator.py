import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.HonestBidder import *


class MarketSimulator:
    def __init__(
        self,
        data: pd.DataFrame = None,
    ):
        self.data = data

    def simulate(
        self,
        bidders: list,
        min_lookback: int = 288,
        rtp_col: str = "rtp",
        dap_col: str = "dap",
        prop_honest: float = 0.8,
    ):
        """
        Simulates the market given a list of bidders

        :param bidders: a list of bidder objects
        :param min_lookback: the minimum number of data points to look back on
        :param rtp_col: the name of the column containing the real-time price data
        :param dap_col: the name of the column containing the day-ahead price data
        :param prop_honest: the proportion of honest bidders in the market
        """
        # copy data for simulation
        self.sim_data = self.data.copy().reset_index(drop=True)

        # determine storage capacity and power capacities
        total_storage = sum([bidder.capacity for bidder in bidders])
        total_capacity = sum([bidder.power_max for bidder in bidders])

        # TODO: initialize honest bidders here

        self.sim_data["scaled_demand"] = (
            self.sim_data.normalized_demand * total_capacity
        )

        self.sim_data = self.sim_data.rename(columns={rtp_col: "rtp", dap_col: "dap"})

        for row_idx in range(min_lookback, len(self.sim_data)):
            print(row_idx)
            # lookback = self.sim_data.iloc[row_idx - min_lookback : row_idx]

    def _transform_bid(self, bid, eff) -> tuple[float, float]:
        return (bid * eff, bid / eff)

    def _clear_market(
        self, bidders: list, demand: float, lookback: pd.DataFrame, rtp: float
    ) -> tuple[list[float], list[float]]:
        """
        Clears the market given a list of bidders, demand, and real-time price

        :param bidders: a list of bidder objects
        :param demand: the total demand in the market
        :param lookback: a pandas DataFrame containing the historical data
        :param rtp: the real-time price
        :return: a tuple containing the total profit and total power cleared in the market
        """
        total_profit = 0
        total_power = 0

        for bidder in bidders:
            bid, power_bounds = bidder.bid(lookback)
            power_lower, power_upper = self._transform_bid(power_bounds, bidder.eff)
            power = self._clamp(bid, power_lower, power_upper)
            total_power += power
            profit = power * rtp
            total_profit += profit
            bidder.step(power, profit)

        return total_profit, total_power
