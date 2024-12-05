import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

class ProbBidClearing:
    """
    This class contains an object to help train an energy market bidder by simulating the probability of bids being accepted.
    """

    def __init__(self, std, risky_mean, conservative_mean):
        """
        Initializes a ProbBidClearing object.

        :param std_fit: Standard deviation of normal distribution based on historical data
        :param max_pdf_fit: Maximum value the normal pdf can achieve with a given std
        :param honest_mean: Mean for the honest bidder (kept constant at 0)
        :param risky_mean: Mean for the risky bidder 
        :param conservative_mean: Mean for the conservative bidder 
        """
        
        ## Standard Deviation
        self.std_fit = std
        self.max_pdf_fit = 1/np.sqrt(2*np.pi*(self.std_fit**2))

        ## Attitudes and Means
        self.attitudes = {
            "honest": 0,
            "risky": risky_mean,
            "conservative": conservative_mean
        }

 
    def norm_prob_clear(self, rtp, bid, attitude, SOC):
        """
        Calculates the probability of a bid being cleared in the electricity market. 
        Based on the training attitude and deviation from real time price (RTP)
    
        :param rtp: current price of electricity ($/MW)
        :param bid: bid submitted to the electricity market ($/MW)
        :param attitude: strategy type employed by the bidder
            'honest': pdf centered around zero deviation from the RTP
            'risky': pdf centered around +std deviation from the RTP (bid high prices)
            'conservative': pdf centered around -std deviation from the RTP (bid low prices)
        :param SOC: current battery state of charge 
        :return: [-1, 0, 1] representing if the bid is accepted to charge, hold, or discharge
        """
        
        mean = self.attitudes[attitude]

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
            

    def custom_norm_prob_clear(self, rtp, bid, SOC, mean, std):
        """
        Calculates the probability of a bid being cleared in the electricity market. 
        Based on custom mean and std for the normal distribution 
    
        :param rtp: current price of electricity ($/MW)
        :param bid: bid submitted to the electricity market ($/MW)
        :param SOC: current battery state of charge 
        :param mean: mean for the normal distribution 
        :param std: std for the normal distribution
        :return: [-1, 0, 1] representing if the bid is accepted to charge, hold, or discharge
        """

        max_pdf_fit = 1/np.sqrt(2*np.pi*(std**2))
        
        # Calculate delta in bid from RTP
        x = rtp - bid
        
        # Compute the PDF, which will act as a threshold for bid acceptance 
        th = norm.pdf(x, loc=mean, scale=std)

        # Draw sample value from uniform distribution ranging over possible pdf values
        s = np.random.uniform(0,max_pdf_fit,1)

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

    
    def visualize_norm_attitudes(self):
        attitudes = self.attitudes
        std_dev = self.std_fit  # Same standard deviation
        
        # Generate x values
        x = np.linspace(- 6*std_dev, 6*std_dev, 1000)
        
        # Plot each distribution
        for label, mean in attitudes.items():
            y = norm.pdf(x, mean, std_dev)
            plt.plot(x, y, label=f'{label.capitalize()} \n(μ={mean}, σ={std_dev})')
            # Add a vertical line for the mean
            plt.axvline(mean, color='red', linestyle='--', linewidth=1.5)
        
        # Customize the plot
        plt.title('Bid Acceptance Distributions with Different Attitudes')
        plt.xlabel(r'$\Delta\lambda \text{ [\$/MW]}$')
        plt.ylabel('Probability Density')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
         
         
    # TO DO: def timevarying_norm_prob_clear(rtp, bid, attitude, SOC, TOD):
    # :param TOD: current TOD. Used to introduce a time-varying aspect to the bid acceptance prob based on known demand patterns






        
