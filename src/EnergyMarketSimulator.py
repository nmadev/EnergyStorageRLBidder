import pandas as pd
import numpy as np


class EnergyMarketSimulator:
    """
    This class simulates an energy market as a test bed for energy storage arbitrage algorithm testing.
    """

    def __init__(self, model, params):
        self.simulation_name = self._safe_lookup(params, "simulation_name", None)
        self.bid_model = model

    def run_simulation(
        self,
        data_path: str = None,
        n_episodes: int = 1,
        start_buffer: int = 0,
        run_description: str = None,
        train: bool = False,
    ) -> pd.DataFrame:
        """
        Loads data from specified path and runs the model through the simulation (either training or evaluation) returning simulation statistics.

        :param data_path: path to load data from
        :param n_episodes: number of episodes
        :param start_buffer: how many periods at the beginning to skip during simulation run (must be >= than maximum lookback window length)
        :param run_description: description of simulation and path to save visualizations to
        :param train: boolean representing whether to train the model or not
        :return: pandas dataframe with simulation statistics
        """
        visualize = True if run_description else False
        fig_save_path = f"./figs/{run_description}"

        # TODO: Load data into a dataframe

        for episode in range(n_episodes):
            # TODO: Reset simulation and build new tracking object

            # TODO: Change to iterate through timesteps one at a time
            for ts in range(100):

                # TODO: Prepare dataset for self.model.step(data=step_data, train=True)
                # TODO: What if we just give the dataset up to that point and let the model architecture take care of filtering/doing whatever?
                pass

    def filter_data(
        self,
        data: pd.DataFrame = None,
        timestamp: pd.Timestamp = None,
        inclusive: bool = False,
    ) -> pd.DataFrame:
        """
        Filters the and copies data to end on the given timestamp

        :param data:
        :param timestamp:
        :param inclusinve:
        :return:
        """
        if data == None:
            raise ValueError
        if timestamp == None:
            raise ValueError

        data_copy = data.copy()
        if inclusive:
            return data_copy[data_copy.index <= timestamp]
        return data_copy[data_copy.index < timestamp]

    def _safe_lookup(dictionary: dict = None, key: str = None, default=None):
        """
        Safely lookup a key in a dictionary and return a default value if not found

        :param dictionary: dictionary containing key-value pairs
        :param key: key to access value from dictionary
        :param default: default value to return
        :return: None if dictionary or key does not exist or if key not in dictionary, otherwise returns value from key in dictionary
        """
        if dictionary == None or key == None or key not in dictionary:
            return default
        return dictionary[key]
