import pandas as pd
import numpy as np

class SimulationHistory:
    """
    This class defines an object to store simulation histories.
    """
    def __init__(
            self,
            description: str = None,
    ):
        self.description = description
        self.history = pd.DataFrame(columns=['DAP', 'RTP', 'bid', 'profit'], 
                                    dtype=float)
        self.history.index = pd.to_datetime([])
        self.history.index.name = 'timestamp'

    def add_to_history(self,
                       ts: pd.Timestamp = None,
                       DAP: float = np.nan,
                       RTP: float = np.nan,
                       bid: float = np.nan,):
        # construct a row from the data
        data_row = [DAP, RTP, bid, bid * RTP]
        
        # Add the data row to self.history at the index ts
        self.history.loc[ts] = data_row
