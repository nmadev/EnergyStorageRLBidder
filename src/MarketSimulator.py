import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from HonestBidder import HonestBidder


class MarketSimulator:
    def __init__(
        self,
        data: pd.DataFrame = None,
    ):
        self.data = data

    def simulate(self, bidders)
