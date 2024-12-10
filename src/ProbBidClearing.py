import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


import imageio
import os
import shutil


pd.options.mode.chained_assignment = None


class ProbBidClearing:
    """
    This class contains an object to help train an energy market bidder by simulating the probability of bids being accepted.
    """

    def __init__(
        self, std=10, risky_mean=15, conservative_mean=-15, alpha=7, window=10
    ):
        """
        Initializes a ProbBidClearing object.

        :param std: Standard deviation of normal distribution based on historical data
        :param honest_mean: Mean for the honest bidder (kept constant at 0)
        :param risky_mean: Mean for the risky bidder
        :param conservative_mean: Mean for the conservative bidder
        :param alpha: Mean-shifting factor for bidding
        :param win: Window size used to compute the roller average demand profile
        """

        ## Standard Deviation
        self.std_fit = std
        self.max_pdf_fit = 1 / np.sqrt(2 * np.pi * (self.std_fit**2))

        ## Attitudes and Means
        self.attitudes = {
            "honest": 0,
            "risky": risky_mean,
            "conservative": conservative_mean,
        }

        ## Attitudes and Scales
        self.attitude_scales = {"honest": 0, "risky": alpha, "conservative": -alpha}
        self.alpha = alpha

        ## Compute Demand Profile
        self.win = window

        # Pull dataset
        self.ENERGY_STORAGE_DATA_PATH = (
            "./src/CAISO-EnergyStorage/src/data/ES_BIDS/CAISO_ES_BIDS.parquet"
        )

        # Read parquet file
        self.STORAGE_DF = pd.read_parquet(self.ENERGY_STORAGE_DATA_PATH)

        # Calculate smoothed profile
        self.avg_profile_rolling = self.avg_demand_profile(self.STORAGE_DF, self.win)

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
        s = np.random.uniform(0, self.max_pdf_fit, 1)

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

        max_pdf_fit = 1 / np.sqrt(2 * np.pi * (std**2))

        # Calculate delta in bid from RTP
        x = rtp - bid

        # Compute the PDF, which will act as a threshold for bid acceptance
        th = norm.pdf(x, loc=mean, scale=std)

        # Draw sample value from uniform distribution ranging over possible pdf values
        s = np.random.uniform(0, max_pdf_fit, 1)

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
        x = np.linspace(-6 * std_dev, 6 * std_dev, 1000)

        # Plot each distribution
        for label, mean in attitudes.items():
            y = norm.pdf(x, mean, std_dev)
            plt.plot(x, y, label=f"{label.capitalize()} \n(μ={mean}, σ={std_dev})")
            # Add a vertical line for the mean
            plt.axvline(mean, color="red", linestyle="--", linewidth=1.5)

        # Customize the plot
        plt.title("Bid Acceptance Distributions with Different Attitudes")
        plt.xlabel(r"$\Delta\lambda \text{ [\$/MW]}$")
        plt.ylabel("Probability Density")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def avg_demand_profile(self, STORAGE_DF, win):
        """
        Calculates the average and smoothed daily demand profile based on moving-average window

        :param STORAGE_DF: raw dataset from CAISO bids
        :param window: window size for rolling moving average
        :return avg_profile_rolling: smoothed demand profile
        """
        # Isolate demand data
        demand_data = STORAGE_DF[["tot_energy_rtpd"]]

        # Extract time of day
        demand_data["time_of_day"] = demand_data.index.time

        # Compute average demand for each 5-minute interval
        avg_profile = demand_data.groupby("time_of_day")["tot_energy_rtpd"].mean()

        # Convert index to string for plotting
        avg_profile.index = avg_profile.index.map(lambda x: x.strftime("%H:%M"))

        # Get last few values
        last_values = avg_profile[-(win - 1) :]

        # Prepend last values to array to get wraparound series
        avg_profile_ext = pd.concat([last_values, avg_profile])

        # Compute rolling average
        avg_profile_ext_rolling = avg_profile_ext.rolling(window=win).mean()

        # Drop first few values (are NaN before window kicks in)
        avg_profile_rolling = avg_profile_ext_rolling[(win - 1) :]

        return avg_profile_rolling

    def visualize_avg_demand_profile(self):

        avg_profile_rolling = self.avg_profile_rolling

        # Calculate the mean
        mean_value = 0  # avg_profile_rolling.mean()

        # Create the figure
        plt.figure(figsize=(10, 6))
        mean_line = plt.axhline(
            y=mean_value, color="r", linestyle="--", label=f"Mean: {mean_value:.2f}"
        )
        (demand,) = plt.plot(avg_profile_rolling.index, avg_profile_rolling.values)
        fill_red = plt.fill_between(
            avg_profile_rolling.index,
            avg_profile_rolling.values,
            mean_value,
            where=(avg_profile_rolling.values < mean_value),
            color="red",
            alpha=0.3,
        )
        fill_green = plt.fill_between(
            avg_profile_rolling.index,
            avg_profile_rolling.values,
            mean_value,
            where=(avg_profile_rolling.values > mean_value),
            color="green",
            alpha=0.3,
        )

        plt.title("Average Smoothed Daily Demand Profile")
        plt.xlabel("Time of Day")
        plt.ylabel("Average Total Energy")
        tick_positions = avg_profile_rolling.index[
            ::12
        ]  # Select every hour entry for the ticks
        plt.xticks(tick_positions, rotation=45)
        plt.xlim("00:00", "23:45")
        plt.grid(True)
        plt.tight_layout()
        plt.legend(
            [demand, fill_red, fill_green],
            ["Demand", "Charge", "Discharge"],
            loc="upper left",
        )
        plt.show()

    def get_demand_profile(self):
        return self.avg_profile_rolling

    def timevarying_norm_prob_clear(self, rtp, bid, attitude, SOC, ts):
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
        :param ts: extract TOD from ts for time-varying bid acceptance prob based on known demand patterns
        :param demand: average smoothed daily regional demand profile
        :return: [-1, 0, 1] representing if the bid is accepted to charge, hold, or discharge
        """
        # Get demand profile
        demand = self.avg_profile_rolling

        # Extract time of day from ts
        TOD = ts.strftime("%H:%M")

        # Extract demand value
        d = demand.loc[TOD]

        # Extract mean from attitude
        mean = self.attitudes[attitude]

        # Calculate delta in bid from RTP
        x = rtp - bid

        # Compute the PDF, which will act as a threshold for bid acceptance
        th = norm.pdf(x, loc=mean, scale=self.std_fit)

        # Draw sample value from uniform distribution ranging over possible pdf values
        s = np.random.uniform(0, self.max_pdf_fit, 1)

        # If the sample is below the threshold, then the bid is accepted
        if s[0] < th:
            # Determine if charge or discharging
            if d > 0 and SOC != 0:
                # Accepted bid, demand > 0 and battery is not empty, discharge
                return 1
            elif d < 0 and SOC != 1:
                # Accepted bid, demand < 0 and battery is not full, charge
                return -1
            else:
                # Our bid was accepted but physical constraints prevent action
                return 0
        else:
            # Bid was not accepted
            return 0

    def meanshift_norm_prob_clear(self, rtp, bid, attitude, SOC, ts):
        """
        Calculates the probability of a bid being cleared in the electricity market.
        Shifts mean based on demand patterns and pre-set attitude

        :param rtp: current price of electricity ($/MW)
        :param bid: bid submitted to the electricity market ($/MW)
        :param attitude: strategy type employed by the bidder
            'honest': pdf centered around zero deviation from the RTP
            'risky': pdf centered around +std deviation from the RTP (bid high prices)
            'conservative': pdf centered around -std deviation from the RTP (bid low prices)
        :param SOC: current battery state of charge
        :param ts: extract TOD from ts for time-varying bid acceptance prob based on known demand patterns
        :param demand: average smoothed daily regional demand profile
        :return: [-1, 0, 1] representing if the bid is accepted to charge, hold, or discharge
        """

        # Get demand profile
        demand = self.avg_profile_rolling

        # Normalize demand profile from [-1, 1]
        min_val = min(demand)
        max_val = max(demand)
        # TODO: Might change to divide by max abs
        demand_norm = 2 * (demand - min_val) / (max_val - min_val) - 1

        # Extract time of day from ts
        TOD = ts.strftime("%H:%M")

        # Extract demand value
        d = demand.loc[TOD]
        d_norm = demand_norm.loc[TOD]

        # Extract scaling factor from attitude
        scale = self.attitude_scales[attitude]

        # Extract mean from attitude and demand pattern
        mean = scale * d_norm

        # Calculate delta in bid from RTP
        x = rtp - bid

        # Compute the PDF, which will act as a threshold for bid acceptance
        th = norm.pdf(x, loc=mean, scale=self.std_fit)

        # Draw sample value from uniform distribution ranging over possible pdf values
        s = np.random.uniform(0, self.max_pdf_fit, 1)

        # If the sample is below the threshold, then the bid is accepted
        if s[0] < th:
            # Determine if charge or discharging
            if d > 0 and SOC != 0:
                # Accepted bid, demand > 0 and battery is not empty, discharge
                return 1
            elif d < 0 and SOC != 1:
                # Accepted bid, demand < 0 and battery is not full, charge
                return -1
            else:
                # Our bid was accepted but physical constraints prevent action
                return 0
        else:
            # Bid was not accepted
            return 0

    def visualize_meanshift(self):
        # Get demand profile
        demand = self.avg_profile_rolling

        # Normalize demand profile
        min_val = min(demand)
        max_val = max(demand)
        # TODO: Might change to divide by max abs
        demand_norm = 2 * (demand - min_val) / (max_val - min_val) - 1

        # Scale factor
        alpha = self.alpha

        # Mean Shifting
        cons_means = -alpha * demand_norm
        risky_means = alpha * demand_norm
        honest_means = 0

        # Create a figure
        fig = plt.figure(figsize=(10, 6))

        # Define the gridspec with height ratios
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])  # 1:2 height ratio

        # Define tick positions
        tick_positions = demand_norm.index[
            ::12
        ]  # Select every hour entry for the ticks
        idx = demand_norm.index

        # Add the demand plot
        ax1 = fig.add_subplot(gs[0, 0])  # First row
        ax1.set_title("Average Smoothed Normalized Daily Demand Profile")
        ax1.plot(idx, demand_norm.values, label="Demand")
        ax1.set_xlabel("Time of Day")
        ax1.set_ylabel("Demand (MW)")
        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels(labels=tick_positions, rotation=45)
        ax1.set_xlim("00:00", "23:45")
        ax1.grid(True)

        # Add the main plot
        ax2 = fig.add_subplot(gs[1, 0])  # Second row
        ax2.set_title("Means of Shifted Probability Clearing Functions")
        ax2.plot(idx, cons_means, label="Conservative", color="r")
        ax2.plot(idx, risky_means, label="Risky", color="b")
        mean_line = plt.axhline(y=honest_means, color="g", label="Honest")
        ax2.set_xlabel("Time of Day")
        ax2.set_ylabel("Means")
        ax2.set_ylim([-10, 10])
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(labels=tick_positions, rotation=45)
        ax2.set_xlim("00:00", "23:45")
        ax2.grid(True)

        # Add an overall legend to the figure in the outer top-right position
        ax1.legend(loc="upper right", bbox_to_anchor=(1.15, 1), title="Overall Legend")

        plt.tight_layout()  # Adjust spacing
        plt.show()

    def visualize_prob_clear(
        self,
        demand_norm,
        attitude_labels,
        attitude_colors,
        attitude_means,
        attitude_stdev,
        attitude_demand_means,
        interest_index=0,
        save_path=None,
    ):
        # TODO: Maybe adjust the colors of the vertical lines + dots for visual clarity
        # Create a figure
        fig = plt.figure(figsize=(16, 8))

        # Define the gridspec with height ratios
        gs = gridspec.GridSpec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1])

        # Define tick positions
        tick_positions = demand_norm.index[
            ::12
        ]  # Select every hour entry for the ticks
        idx = demand_norm.index

        # Add the demand plot
        ax1 = fig.add_subplot(gs[0, 0])  # First row, first column
        ax1.set_title("Average Smoothed Normalized Daily Demand Profile")
        ax1.plot(idx, demand_norm.values, label="Demand")
        ax1.set_xlabel("Time of Day")
        ax1.set_ylabel("Demand (MW)")
        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels(labels=tick_positions, rotation=45)
        ax1.axhline(y=0, color="black")
        ax1.scatter(
            x=[idx[interest_index]],
            y=[demand_norm.values[interest_index]],
            color="C0",
        )
        ax1.plot(
            [idx[interest_index], idx[interest_index]],
            [0, demand_norm.values[interest_index]],
            color="C0",
        )

        ax1.set_xlim("00:00", "23:45")
        ax1.grid(True)

        # Add the main plot
        ax2 = fig.add_subplot(gs[1, 0])  # Second row, first column
        ax2.set_title("Means of Shifted Probability Clearing Functions")
        for i in range(3):
            ax2.plot(
                idx,
                attitude_demand_means[i],
                label=attitude_labels[i],
                color=attitude_colors[i],
            )
        ax2.set_xlabel("Time of Day")
        ax2.set_ylabel("Means")
        ax2.set_ylim([-17, 17])
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(labels=tick_positions, rotation=45)
        for i in [2, 1, 0]:
            ax2.scatter(
                x=[idx[interest_index]],
                y=[attitude_demand_means[i][interest_index]],
                color=attitude_colors[i],
            )
            ax2.plot(
                [idx[interest_index], idx[interest_index]],
                [0, attitude_demand_means[i][interest_index]],
                color=attitude_colors[i],
            )

        ax2.set_xlim("00:00", "23:45")
        ax2.grid(True)

        # Add the combined plot to the right
        ax3 = fig.add_subplot(gs[:, 1])  # Span both rows, second column
        ax3.set_title("Bid Acceptance Probability Functions")
        ax3.set_xlim([-50, 50])
        ax3.set_ylim([0, 0.0425])
        # ax3.plot(idx, demand_norm.values, label="Demand", linestyle="--")
        for i in range(3):
            mean = attitude_demand_means[i][interest_index]
            std = attitude_stdev[i]
            x = np.linspace(mean - 3 * std, mean + 3 * std, 100)
            y = norm.pdf(x, mean, std)
            ax3.plot(
                x,
                y,
                label=f"{attitude_labels[i]} (μ={mean:.2f}, σ={std})",
                color=attitude_colors[i],
            )
            ax3.axvline(mean, color=attitude_colors[i], linestyle="--")

        ax3.set_xlabel("Bid Deviation from RTP ($/MW)")
        ax3.set_ylabel("Probability Density")
        ax3.legend(loc="upper right")
        ax3.grid(True)

        plt.tight_layout()  # Adjust spacing
        if save_path:
            plt.savefig(save_path, dpi=100)
        plt.close()

    def animate_prob_clear(self, prob_function, label=""):
        # Get demand profile
        demand = self.avg_profile_rolling

        os.makedirs("./images/ProbBidClearing/", exist_ok=True)
        os.makedirs("./animations/", exist_ok=True)

        attitude_labels = ["Neutral", "Risky", "Conservative"]
        attitude_colors = ["g", "b", "r"]
        attitude_means = [0.0, 15, -15]
        attitude_stdev = [10.0] * 3

        demand_norm = demand / np.max(np.abs(demand))

        attitude_demand_means = [[], [], []]
        for norm in demand_norm:
            for i in range(3):
                attitude_demand_means[i].append(prob_function(norm, attitude_means[i]))

        for i in range(len(demand_norm)):
            self.visualize_prob_clear(
                demand_norm,
                attitude_labels,
                attitude_colors,
                attitude_means,
                attitude_stdev,
                attitude_demand_means,
                i,
                save_path=f"./images/ProbBidClearing/ProbBidClearing_{label}_{i:03d}.png",
            )

    def write_animation(self, label="", fps=24):
        # load all files in order and create a gif
        images = []
        for filename in sorted(os.listdir("./images/ProbBidClearing/")):
            print("Loading image: ", filename, end="\r")
            images.append(imageio.imread(f"./images/ProbBidClearing/{filename}"))
        print("Saving...")
        imageio.mimsave(
            f"./animations/ProbBidClearing_animation_{label}.gif",
            images,
            fps=fps,
            format="GIF",
            loop=0,
        )
        del(images)

    def clear_images(self):
        # clear all images
        shutil.rmtree("./images/ProbBidClearing/")
        os.makedirs("./images/ProbBidClearing/")
