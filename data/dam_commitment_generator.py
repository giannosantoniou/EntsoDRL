"""
DAM Commitment Generator Module

This module provides swappable generators for DAM (Day-Ahead Market) commitments.
For training, use FeasibleDAMGenerator which produces battery-feasible commitments.
For production, replace with your price prediction system's output.

Usage:
    from data.dam_commitment_generator import FeasibleDAMGenerator

    generator = FeasibleDAMGenerator(capacity_mwh=50, max_power_mw=50)
    df['dam_commitment'] = generator.generate(df)
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Tuple


class DAMCommitmentGenerator(ABC):
    """Abstract base class for DAM commitment generators."""

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.Series:
        """Generate DAM commitments for each row in the dataframe."""
        pass


class FeasibleDAMGenerator(DAMCommitmentGenerator):
    """
    Generates battery-feasible DAM commitments based on price signals.

    Strategy:
    - Discharge (positive commitment) when price > median
    - Charge (negative commitment) when price < median
    - Commitments are sized to respect SoC constraints
    - Conservative sizing leaves room for DRL optimization
    """

    def __init__(
        self,
        capacity_mwh: float = 146.0,
        max_power_mw: float = 30.0,
        efficiency: float = 0.94,
        initial_soc: float = 0.5,
        min_soc: float = 0.15,  # I increased buffer (was 0.10)
        max_soc: float = 0.85,  # I decreased ceiling (was 0.90)
        commitment_ratio: float = 0.35,  # I reduced to 35% (was 60%) - much more conservative!
        lookahead_hours: int = 24,
    ):
        self.capacity_mwh = capacity_mwh
        self.max_power_mw = max_power_mw
        self.efficiency = efficiency
        self.initial_soc = initial_soc
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.commitment_ratio = commitment_ratio
        self.lookahead_hours = lookahead_hours

    def generate(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate feasible DAM commitments based on price signals.

        Args:
            df: DataFrame with 'price' column

        Returns:
            Series of DAM commitments (positive = discharge, negative = charge)
        """
        df_lower = df.copy()
        df_lower.columns = [c.lower() for c in df_lower.columns]

        prices = df_lower['price'].values
        n_steps = len(prices)

        # Calculate rolling median price for decision making
        window = min(self.lookahead_hours * 7, n_steps // 10)  # ~1 week window
        price_median = pd.Series(prices).rolling(window=window, center=True, min_periods=1).median().values

        # Calculate price percentiles for commitment sizing
        price_percentile = pd.Series(prices).rolling(window=window, center=True, min_periods=1).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-6) if len(x) > 0 else 0.5
        ).fillna(0.5).values

        commitments = np.zeros(n_steps)
        simulated_soc = self.initial_soc

        eff_sqrt = np.sqrt(self.efficiency)

        for i in range(n_steps):
            price = prices[i]
            median = price_median[i]
            percentile = price_percentile[i]

            # Calculate available capacity
            available_discharge_mwh = max(0, (simulated_soc - self.min_soc) * self.capacity_mwh)
            available_charge_mwh = max(0, (self.max_soc - simulated_soc) * self.capacity_mwh)

            # Convert to MW (1 hour intervals assumed)
            max_discharge_mw = min(self.max_power_mw, available_discharge_mwh * eff_sqrt)
            max_charge_mw = min(self.max_power_mw, available_charge_mwh / eff_sqrt)

            if price > median * 1.1:  # Price significantly above median
                # Discharge - amount proportional to how far above median
                intensity = min(1.0, (price / median - 1.0) * 2)  # 0 to 1
                commitment = max_discharge_mw * self.commitment_ratio * intensity
            elif price < median * 0.9:  # Price significantly below median
                # Charge - amount proportional to how far below median
                intensity = min(1.0, (1.0 - price / median) * 2)  # 0 to 1
                commitment = -max_charge_mw * self.commitment_ratio * intensity
            else:
                # Price near median - small or no commitment
                commitment = 0.0

            # Apply some randomness for variety (±20%)
            noise = np.random.uniform(0.8, 1.2)
            commitment = commitment * noise

            # Clamp to feasible range
            commitment = np.clip(commitment, -max_charge_mw, max_discharge_mw)

            commitments[i] = commitment

            # Update simulated SoC for next iteration
            if commitment > 0:  # Discharging
                energy_out = commitment / eff_sqrt
                simulated_soc -= energy_out / self.capacity_mwh
            else:  # Charging
                energy_in = abs(commitment) * eff_sqrt
                simulated_soc += energy_in / self.capacity_mwh

            # I fixed: was [0.0, 1.0], now uses actual min/max SoC limits
            simulated_soc = np.clip(simulated_soc, self.min_soc, self.max_soc)

        return pd.Series(commitments, index=df.index, name='dam_commitment')


class SmartArbitrageDAMGenerator(DAMCommitmentGenerator):
    """
    Simple and correct DAM commitment generator.

    KEY PRINCIPLE: Only commit to what you CAN deliver!

    For each hour:
    1. Look at the price relative to rolling median
    2. If price is high → want to SELL (discharge)
    3. If price is low → want to BUY (charge)
    4. Before committing: "Do I have enough SoC for this?"
    5. Only commit if yes, skip if no

    This is simple, correct, and lets the AGENT learn strategic planning
    through 12-hour lookahead during training.
    """

    def __init__(
        self,
        capacity_mwh: float = 146.0,
        max_power_mw: float = 30.0,
        efficiency: float = 0.94,
        initial_soc: float = 0.5,
        min_soc: float = 0.10,  # I use full range - agent will learn to manage
        max_soc: float = 0.90,
        commitment_ratio: float = 0.50,  # I commit to 50% of max power
        price_threshold_high: float = 1.15,  # I sell when price > 115% of median
        price_threshold_low: float = 0.85,   # I buy when price < 85% of median
    ):
        self.capacity_mwh = capacity_mwh
        self.max_power_mw = max_power_mw
        self.efficiency = efficiency
        self.initial_soc = initial_soc
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.commitment_ratio = commitment_ratio
        self.price_threshold_high = price_threshold_high
        self.price_threshold_low = price_threshold_low

    def generate(self, df: pd.DataFrame) -> pd.Series:
        """
        I generate DAM commitments with simple "do I have enough?" logic.

        For each step:
        - If price is high and I have enough SoC → commit to discharge
        - If price is low and I have room to charge → commit to charge
        - Otherwise → no commitment

        The agent will learn when to ACTUALLY charge/discharge through
        its 12-hour lookahead. I just create feasible commitments.
        """
        df_work = df.copy()
        df_work.columns = [c.lower() for c in df_work.columns]

        prices = df_work['price'].values
        n_steps = len(prices)

        # I calculate rolling median for price reference (168h = 1 week window)
        window = min(168, n_steps // 5)
        price_median = pd.Series(prices).rolling(
            window=window, center=True, min_periods=1
        ).median().values

        commitments = np.zeros(n_steps)
        simulated_soc = self.initial_soc
        eff_sqrt = np.sqrt(self.efficiency)

        for i in range(n_steps):
            price = prices[i]
            median = price_median[i]

            # I calculate what I CAN do (physical limits)
            available_discharge_mwh = max(0, (simulated_soc - self.min_soc) * self.capacity_mwh)
            available_charge_mwh = max(0, (self.max_soc - simulated_soc) * self.capacity_mwh)

            max_discharge_mw = min(self.max_power_mw, available_discharge_mwh * eff_sqrt)
            max_charge_mw = min(self.max_power_mw, available_charge_mwh / eff_sqrt)

            commitment = 0.0

            # HIGH PRICE: Do I want to sell? Do I have enough?
            if price > median * self.price_threshold_high:
                if max_discharge_mw > 1.0:  # I have at least 1MW to sell
                    # I commit to a portion of what I can deliver
                    intensity = min(1.0, (price / median - 1.0) * 3)  # I scale by price deviation
                    commitment = max_discharge_mw * self.commitment_ratio * intensity

            # LOW PRICE: Do I want to buy? Do I have room?
            elif price < median * self.price_threshold_low:
                if max_charge_mw > 1.0:  # I have at least 1MW room
                    # I commit to a portion of what I can accept
                    intensity = min(1.0, (1.0 - price / median) * 3)  # I scale by price deviation
                    commitment = -max_charge_mw * self.commitment_ratio * intensity

            # I add small noise for variety (±10%)
            if abs(commitment) > 0.5:
                noise = np.random.uniform(0.9, 1.1)
                commitment = commitment * noise

            # I clamp to feasible range (should already be, but safety check)
            commitment = np.clip(commitment, -max_charge_mw, max_discharge_mw)
            commitments[i] = commitment

            # I update simulated SoC for next iteration
            if commitment > 0:  # Discharging
                energy_out = commitment / eff_sqrt
                simulated_soc -= energy_out / self.capacity_mwh
            elif commitment < 0:  # Charging
                energy_in = abs(commitment) * eff_sqrt
                simulated_soc += energy_in / self.capacity_mwh

            simulated_soc = np.clip(simulated_soc, 0.05, 0.95)

        return pd.Series(commitments, index=df.index, name='dam_commitment')


class OptimalLookaheadDAMGenerator(DAMCommitmentGenerator):
    """
    I created this generator to solve the optimal arbitrage problem exactly.

    Instead of heuristics, I use a simple dynamic programming approach:
    - For each 24h window, I find the optimal charge/discharge schedule
    - I maximize profit while respecting SoC and power constraints
    - This mimics what a perfect DAM trader would submit

    Note: This is computationally heavier but produces more realistic commitments.
    """

    def __init__(
        self,
        capacity_mwh: float = 146.0,
        max_power_mw: float = 30.0,
        efficiency: float = 0.94,
        initial_soc: float = 0.5,
        min_soc: float = 0.10,
        max_soc: float = 0.90,
        commitment_ratio: float = 0.60,  # Higher ratio since I know exactly when to commit
        lookahead_hours: int = 24,
        n_soc_bins: int = 20,  # Discretization for DP
    ):
        self.capacity_mwh = capacity_mwh
        self.max_power_mw = max_power_mw
        self.efficiency = efficiency
        self.initial_soc = initial_soc
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.commitment_ratio = commitment_ratio
        self.lookahead_hours = lookahead_hours
        self.n_soc_bins = n_soc_bins

    def _solve_window_dp(
        self,
        prices: np.ndarray,
        start_soc: float
    ) -> Tuple[np.ndarray, float]:
        """
        I solve optimal trading for a window using dynamic programming.

        State: SoC (discretized)
        Action: charge/discharge amount
        Objective: maximize profit
        """
        n_hours = len(prices)
        eff_sqrt = np.sqrt(self.efficiency)

        # I discretize SoC
        soc_values = np.linspace(self.min_soc, self.max_soc, self.n_soc_bins)

        # I find closest starting SoC bin
        start_bin = np.argmin(np.abs(soc_values - start_soc))

        # Value function: V[t, s] = max profit from time t with SoC bin s
        V = np.full((n_hours + 1, self.n_soc_bins), -np.inf)
        policy = np.zeros((n_hours, self.n_soc_bins))  # Best action for each state

        # I initialize terminal values (any ending SoC is ok)
        V[n_hours, :] = 0.0

        # I backward induct
        for t in range(n_hours - 1, -1, -1):
            price = prices[t]

            for s_idx, soc in enumerate(soc_values):
                best_value = -np.inf
                best_action = 0.0

                # I try different actions
                available_discharge = min(
                    self.max_power_mw,
                    (soc - self.min_soc) * self.capacity_mwh * eff_sqrt
                )
                available_charge = min(
                    self.max_power_mw,
                    (self.max_soc - soc) * self.capacity_mwh / eff_sqrt
                )

                # I discretize actions (0%, 25%, 50%, 75%, 100% of max)
                discharge_options = [0, 0.25, 0.5, 0.75, 1.0]

                for d_frac in discharge_options:
                    discharge = available_discharge * d_frac

                    # New SoC after discharge
                    energy_out = discharge / eff_sqrt
                    new_soc = soc - energy_out / self.capacity_mwh

                    if new_soc >= self.min_soc - 0.01:
                        profit = discharge * price
                        new_s_idx = np.argmin(np.abs(soc_values - new_soc))
                        value = profit + V[t + 1, new_s_idx]

                        if value > best_value:
                            best_value = value
                            best_action = discharge

                for c_frac in discharge_options:
                    charge = available_charge * c_frac

                    # New SoC after charge
                    energy_in = charge * eff_sqrt
                    new_soc = soc + energy_in / self.capacity_mwh

                    if new_soc <= self.max_soc + 0.01:
                        profit = -charge * price  # Negative because we pay to charge
                        new_s_idx = np.argmin(np.abs(soc_values - new_soc))
                        value = profit + V[t + 1, new_s_idx]

                        if value > best_value:
                            best_value = value
                            best_action = -charge  # Negative indicates charging

                V[t, s_idx] = best_value
                policy[t, s_idx] = best_action

        # I extract optimal actions by forward simulation
        optimal_actions = np.zeros(n_hours)
        current_soc = start_soc

        for t in range(n_hours):
            s_idx = np.argmin(np.abs(soc_values - current_soc))
            action = policy[t, s_idx]

            # I apply commitment ratio (keep some flexibility for DRL)
            action = action * self.commitment_ratio

            optimal_actions[t] = action

            # Update SoC
            if action > 0:
                energy_out = action / eff_sqrt
                current_soc -= energy_out / self.capacity_mwh
            else:
                energy_in = abs(action) * eff_sqrt
                current_soc += energy_in / self.capacity_mwh

            current_soc = np.clip(current_soc, self.min_soc, self.max_soc)

        return optimal_actions, current_soc

    def generate(self, df: pd.DataFrame) -> pd.Series:
        """I generate commitments using rolling window DP optimization."""
        df_work = df.copy()
        df_work.columns = [c.lower() for c in df_work.columns]

        prices = df_work['price'].values
        n_steps = len(prices)

        commitments = np.zeros(n_steps)
        current_soc = self.initial_soc

        # I process in windows
        window_size = self.lookahead_hours
        i = 0

        while i < n_steps:
            end_i = min(i + window_size, n_steps)
            window_prices = prices[i:end_i]

            # I solve optimal trading for this window
            window_actions, final_soc = self._solve_window_dp(window_prices, current_soc)

            # I only commit to first few hours (re-optimize as we go)
            commit_hours = min(window_size // 4, len(window_actions))

            for j in range(commit_hours):
                if i + j < n_steps:
                    # I add small noise for variety
                    noise = np.random.uniform(0.9, 1.1)
                    commitments[i + j] = window_actions[j] * noise

            # I update SoC based on committed actions
            for j in range(commit_hours):
                action = commitments[i + j]
                eff_sqrt = np.sqrt(self.efficiency)

                if action > 0:
                    energy_out = action / eff_sqrt
                    current_soc -= energy_out / self.capacity_mwh
                else:
                    energy_in = abs(action) * eff_sqrt
                    current_soc += energy_in / self.capacity_mwh

                current_soc = np.clip(current_soc, self.min_soc, self.max_soc)

            i += commit_hours

        return pd.Series(commitments, index=df.index, name='dam_commitment')


class SimpleRuleBasedGenerator(DAMCommitmentGenerator):
    """
    Simple rule-based generator using hour-of-day patterns.
    Good for baseline comparison.
    """
    
    def __init__(
        self,
        max_power_mw: float = 30.0,
        commitment_ratio: float = 0.4,
    ):
        self.max_power_mw = max_power_mw
        self.commitment_ratio = commitment_ratio
        
    def generate(self, df: pd.DataFrame) -> pd.Series:
        """Generate commitments based on typical daily patterns."""
        
        hours = df.index.hour if hasattr(df.index, 'hour') else pd.to_datetime(df.index).hour
        
        # Peak hours (17:00-21:00): Discharge
        # Off-peak hours (02:00-06:00): Charge
        # Other times: Small or no commitment
        
        commitments = np.zeros(len(df))
        
        for i, hour in enumerate(hours):
            if 17 <= hour <= 21:  # Evening peak
                commitments[i] = self.max_power_mw * self.commitment_ratio * np.random.uniform(0.8, 1.0)
            elif 2 <= hour <= 6:  # Night off-peak
                commitments[i] = -self.max_power_mw * self.commitment_ratio * np.random.uniform(0.6, 0.8)
            elif 10 <= hour <= 14:  # Midday (solar peak, low prices)
                commitments[i] = -self.max_power_mw * self.commitment_ratio * np.random.uniform(0.3, 0.5)
            else:
                commitments[i] = 0.0
                
        return pd.Series(commitments, index=df.index, name='dam_commitment')


class RandomGenerator(DAMCommitmentGenerator):
    """
    Random commitment generator for baseline/stress testing.
    WARNING: May produce infeasible commitments!
    """

    def __init__(self, max_power_mw: float = 30.0, sparsity: float = 0.7):
        self.max_power_mw = max_power_mw
        self.sparsity = sparsity  # Probability of zero commitment
        
    def generate(self, df: pd.DataFrame) -> pd.Series:
        """Generate random commitments."""
        n = len(df)
        commitments = np.random.uniform(-self.max_power_mw, self.max_power_mw, n)
        
        # Apply sparsity (many zeros)
        mask = np.random.random(n) < self.sparsity
        commitments[mask] = 0.0
        
        return pd.Series(commitments, index=df.index, name='dam_commitment')


def regenerate_dam_commitments(
    df: pd.DataFrame,
    generator: Optional[DAMCommitmentGenerator] = None,
    battery_params: Optional[Dict] = None,
    use_smart_generator: bool = True,  # I default to the new smart generator
) -> pd.DataFrame:
    """
    Convenience function to regenerate DAM commitments in a dataframe.

    Args:
        df: Original dataframe
        generator: DAMCommitmentGenerator instance (default: SmartArbitrageDAMGenerator)
        battery_params: Optional dict with capacity_mwh, max_power_mw, efficiency
        use_smart_generator: If True (default), use SmartArbitrageDAMGenerator; else FeasibleDAMGenerator

    Returns:
        DataFrame with updated dam_commitment column
    """
    if generator is None:
        params = battery_params or {}

        if use_smart_generator:
            # I use the improved smart arbitrage generator by default
            generator = SmartArbitrageDAMGenerator(
                capacity_mwh=params.get('capacity_mwh', 146.0),
                max_power_mw=params.get('max_discharge_mw', 30.0),
                efficiency=params.get('efficiency', 0.94),
            )
        else:
            # Fallback to simple generator
            generator = FeasibleDAMGenerator(
                capacity_mwh=params.get('capacity_mwh', 146.0),
                max_power_mw=params.get('max_discharge_mw', 30.0),
                efficiency=params.get('efficiency', 0.94),
            )

    df = df.copy()
    df['dam_commitment'] = generator.generate(df)

    return df


if __name__ == "__main__":
    # I compare all generators
    import os
    from data_loader import load_historical_data

    # I use absolute path to find the data file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "feasible_data_with_balancing.csv")

    print("Loading data...")
    df = load_historical_data(data_path)

    print("\nOriginal DAM commitments:")
    print(df['dam_commitment'].describe())

    # I test all three generators
    generators = {
        "FeasibleDAMGenerator (old)": FeasibleDAMGenerator(
            capacity_mwh=146, max_power_mw=30, efficiency=0.94
        ),
        "SmartArbitrageDAMGenerator (new)": SmartArbitrageDAMGenerator(
            capacity_mwh=146, max_power_mw=30, efficiency=0.94
        ),
        "OptimalLookaheadDAMGenerator (DP)": OptimalLookaheadDAMGenerator(
            capacity_mwh=146, max_power_mw=30, efficiency=0.94
        ),
    }

    prices = df['price'].values

    for name, gen in generators.items():
        print(f"\n{'='*60}")
        print(f"{name}")
        print("="*60)

        commitments = gen.generate(df)

        print(f"\nStatistics:")
        print(commitments.describe())

        print(f"\n% of time discharging: {(commitments > 0).mean()*100:.1f}%")
        print(f"% of time charging: {(commitments < 0).mean()*100:.1f}%")
        print(f"% of time idle: {(commitments == 0).mean()*100:.1f}%")

        # I calculate theoretical profit (ignoring efficiency for simplicity)
        discharge_profit = (commitments[commitments > 0] * prices[commitments > 0]).sum()
        charge_cost = (-commitments[commitments < 0] * prices[commitments < 0]).sum()
        net_profit = discharge_profit - charge_cost

        print(f"\nTheoretical P&L (ignoring efficiency):")
        print(f"  Discharge revenue: {discharge_profit:,.0f} EUR")
        print(f"  Charge cost:       {charge_cost:,.0f} EUR")
        print(f"  Net profit:        {net_profit:,.0f} EUR")

        # I check if commitments align with price patterns
        high_price_mask = prices > np.percentile(prices, 75)
        low_price_mask = prices < np.percentile(prices, 25)

        discharge_at_high = (commitments[high_price_mask] > 0).mean() * 100
        charge_at_low = (commitments[low_price_mask] < 0).mean() * 100

        print(f"\nSmart trading check:")
        print(f"  Discharging during high prices (top 25%): {discharge_at_high:.1f}%")
        print(f"  Charging during low prices (bottom 25%):  {charge_at_low:.1f}%")
