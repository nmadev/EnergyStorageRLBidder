import numpy as np
import pandas as pd
from scipy.stats import norm

class ProbBidClearing:
    """
    This class contains an object to help train an energy market bidder by simulating the probability of bids being accepted.
    """

    def __init__(self):
        """
        Initializes a ProbBidClearing object.

        :param data: Historical demand data to inform the bidding probabilities 
        :param std: Standard deviation of normal distribution based on historical data
        :param mean_scale: Used to shift mean around when using attitude based bidding
        :param max_pdf: Maximum value the normal pdf can achieve with a given std
        """
        self.std_fit = 15.39
        self.max_pdf_fit = 1/np.sqrt(2*np.pi*(self.std_fit**2))
        self.honest_mean = 0
        self.risky_mean = self.std_fit
        self.conservative_mean = -self.std_fit

 
    def norm_prob_clear(self, rtp, bid, attitude, SOC):
        """
        Calculates the probability of a bid being cleared in the electricity market. 
        Based on the training attitude and deviation from real time price (RTP)
    
        :param rtp: current price of electricity ($/MW)
        :param bid: bid submitted to the electricity market ($/MW)
        :param attitude: strategy type employed by the bidder
            'honest': pdf centered around zero deviation from the RTP
            'risky': pdf centered around +std/2 deviation from the RTP (bid high prices)
            'conservative': pdf centered around -std/2 deviation from the RTP (bid low prices)
        :param SOC: current battery state of charge 
        :return: [-1, 0, 1] representing if the bid is accepted to charge, hold, or discharge
        """
        
        if attitude == "honest":
            mean = self.honest_mean
        elif attitude == "risky":
            mean = self.risk_mean
        elif attitude == "conservative":
            mean = -self.conservative_mean

        # Calculate delta in bid from RTP
        x = rtp - bid
        
        # Compute the PDF, which will act as a threshold for bid acceptance 
        th = norm.pdf(x, loc=mean, scale=self.std_fit)

        # Draw sample value from uniform distribution ranging over possible pdf values
        s = np.random.uniform(0,self.max_pdf_fit,1)

        # If the sample is below the threshold, then the bid is accepted 
        if s[0] < th: 
            # Determine if charge or discharging 
            if x < 0 and SOC != 0:
                # Accepted bid is above RTP and battery is not empty, discharge
                return 1
            elif x > 0 and SOC != 1:
                # Accepted bid is below RTP and battery is not full, charge
                return -1
            else:
                # Our bid was accepted but physical constraints prevent action 
                return 0
        else: 
            # Bid was not accepted
            return 0
            


    # TO DO: def custom_norm_prob_clear(rtp, bid, SOC, mean, std):

    # TO DO: def timevarying_norm_prob_clear(rtp, bid, attitude, SOC, TOD):
    # :param TOD: current TOD. Used to introduce a time-varying aspect to the bid acceptance prob based on known demand patterns

    # TO DO: def visualize_norm_attitudes(self)

    # TO DO: Inverted absolute value function 






        
