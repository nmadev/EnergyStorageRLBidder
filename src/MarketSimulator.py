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
        self.clearing_prices = []

    def simulate(
        self,
        bidders: list,
        min_lookback: int = 288,
        rtp_col: str = "rtp",
        dap_col: str = "dap",
        prop_honest: float = 0.8,
        max_steps: int = -1,
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

        step_value = 0
        for row_idx in range(min_lookback, len(self.sim_data)):
            # TODO: make sure to randomly sort the bidders before clearing the market (maybe todo in the future)
            lookback = self.sim_data.iloc[:row_idx]
            demand = lookback.scaled_demand.iloc[-1]
            rtp = lookback.rtp.iloc[-1]
            dap = lookback.dap.iloc[-1]

            bids = []
            for bidder in bidders:
                bids.append(bidder.bid(lookback))

            # clear market
            cleared_bids = self._clear_market(bidders, bids, demand, lookback, rtp)

            # update state of bidders
            for cleared_bid, bidder in zip(cleared_bids, bidders):
                power, profit = cleared_bid
                bidder.step(power, profit)
            step_value += 1
            if max_steps > 0 and step_value >= max_steps:
                break

    def _transform_bid(self, bid, eff) -> tuple[float, float]:
        return (bid * eff, bid / eff)

    def _clear_market(
        self,
        bidders: list,
        bids: list,
        demand: float,
        lookback: pd.DataFrame,
        rtp: float,
    ) -> list[tuple[float, float]]:
        """
        Clears the market given a list of bidders, bid, overall demand, and real-time price

        :param bidders: a list of bidder objects
        :param bids: a list of bids from each bidder object
        :param demand: the total demand in the market
        :param lookback: a pandas DataFrame containing the historical data
        :param rtp: the real-time price
        :return: a tuple containing the total profit and total power cleared in the market
        """

        # no clearing if no demand
        base_bids = [(0, 0)] * len(bidders)
        if demand == 0:
            return base_bids

        effs = np.array([bidder.eff for bidder in bidders])
        bid_values = np.array([bid[0] for bid in bids])
        bid_powers = np.array([bid[1] for bid in bids])
        zipped_bids = list(
            zip(
                self._transform_bid(bid_values, effs),
                bid_powers,
                effs,
                range(len(bidders)),
            )
        )

        # check charge or discharge
        sign = abs(demand) / demand

        # sort bids on value
        if sign < 0:
            # sort bids in descending order
            zipped_bids = sorted(zipped_bids, key=lambda x: x[0][0], reverse=True)
        else:
            # sort bids in ascending order
            zipped_bids = sorted(zipped_bids, key=lambda x: x[0][1])

        # determine clearing price
        cleared_bids = []
        cleared_demand = 0
        clearing_price = np.nan

        for bid_bounds, power_bounds, effs, idx in zipped_bids:
            available_power = (power_bounds[1] if sign > 0 else power_bounds[0]) * sign
            cleared_power = min((demand - cleared_demand) * sign, available_power)
            cleared_demand += cleared_power * sign
            cleared_bids.append((cleared_power, 0))

            if cleared_demand == demand and clearing_price != np.nan:
                clearing_price = bid_bounds[0] if sign > 0 else bid_bounds[1]

            if cleared_demand >= demand:
                cleared_bids.append((0, 0))

        for i, (bid_bounds, power_bounds, effs, idx) in enumerate(zipped_bids):
            base_bids[idx] = (
                cleared_bids[i][0] * sign,
                cleared_bids[i][0] * sign * clearing_price,
            )

        print(demand, clearing_price)

        # sort back to original order
        return base_bids
