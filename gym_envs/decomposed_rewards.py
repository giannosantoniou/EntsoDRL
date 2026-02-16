"""
Decomposed Market-Specific Reward Calculators

I implement 4 separate reward calculators, one per market:
- DAMRewardCalculator: Revenue from hourly commitments using FORECAST prices
- AFRRRewardCalculator: Capacity payments + energy activation revenue
- IntraDayRewardCalculator: XBID bid/ask spread trading revenue
- MFRRRewardCalculator: Free Bid activation revenue

Key Design Decisions:
1. Each calculator has its own reward_scale (0.001) for NN stability
2. Degradation cost is configurable and ramps via curriculum (0 -> 15 EUR/MWh)
3. Returns (total_reward, info_dict) for TensorBoard logging
4. Penalties are market-specific (e.g., DAM violation vs aFRR non-response)
"""

import numpy as np
from typing import Dict, Tuple


class DAMRewardCalculator:
    """
    I calculate rewards for the DAM (Day-Ahead Market) model.

    Revenue comes from hourly commitments at FORECAST prices (not actuals).
    The agent decides 24 hourly commitments at D-1 12:00.
    """

    def __init__(
        self,
        degradation_cost_per_mwh: float = 15.0,
        reward_scale: float = 0.001,
        timing_weight: float = 1.0,
        flexibility_bonus_weight: float = 5.0,
        soc_end_penalty: float = 50.0,
    ):
        self.degradation_cost = degradation_cost_per_mwh
        self.reward_scale = reward_scale
        self.timing_weight = timing_weight
        self.flexibility_bonus_weight = flexibility_bonus_weight
        self.soc_end_penalty = soc_end_penalty

    def calculate(
        self,
        action_mw: float,
        forecast_price: float,
        soc: float,
        capacity_mwh: float,
        hour: int,
        is_last_step: bool = False,
        total_planned_revenue: float = 0.0,
        remaining_capacity_frac: float = 1.0,
        daily_cycles: float = 0.0,
        max_daily_cycles: float = 2.0,
    ) -> Tuple[float, Dict]:
        """
        I calculate the step reward for a DAM commitment decision.

        Args:
            action_mw: Committed MW (+ = sell, - = buy)
            forecast_price: Forecast DAM price for this hour (NOT actual)
            soc: Current SoC after accounting for prior hours
            capacity_mwh: Battery capacity in MWh
            hour: Hour index (0-23)
            is_last_step: True if hour == 23
            total_planned_revenue: Sum of schedule * forecast prices (at end)
            remaining_capacity_frac: Fraction of capacity left for downstream
            daily_cycles: Current daily cycle count
            max_daily_cycles: Maximum daily cycles

        Returns:
            (reward, info_dict) tuple
        """
        components = {}

        # I calculate step revenue from the commitment at forecast price
        step_revenue = action_mw * forecast_price * 1.0  # 1 hour duration
        components['step_revenue'] = step_revenue

        # I calculate degradation cost from cycling
        energy_mwh = abs(action_mw) * 1.0  # 1 hour
        degradation = energy_mwh * self.degradation_cost / capacity_mwh
        components['degradation'] = degradation

        # I reward price alignment: sell at peak, buy at valley
        timing_bonus = 0.0
        if forecast_price > 0:
            # I normalize price to [-1, 1] range using heuristic thresholds
            price_z = (forecast_price - 80.0) / 40.0  # ~80 EUR mean, ~40 std
            price_z = np.clip(price_z, -2.0, 2.0)

            if action_mw > 0.1:  # selling
                timing_bonus = price_z * abs(action_mw) * self.timing_weight * 0.1
            elif action_mw < -0.1:  # buying
                timing_bonus = -price_z * abs(action_mw) * self.timing_weight * 0.1
        components['timing_bonus'] = timing_bonus

        # I add episode-end bonuses at step 23
        end_bonus = 0.0
        if is_last_step:
            # I reward total planned revenue
            end_bonus += total_planned_revenue * 0.01

            # I reward leaving headroom for downstream markets
            flexibility = remaining_capacity_frac * self.flexibility_bonus_weight
            end_bonus += flexibility

            # I penalize extreme SoC at end of schedule
            if soc < 0.15 or soc > 0.90:
                end_bonus -= self.soc_end_penalty

            # I penalize excessive cycling
            if daily_cycles > max_daily_cycles * 0.8:
                end_bonus -= 10.0 * (daily_cycles - max_daily_cycles * 0.8)

        components['end_bonus'] = end_bonus

        total = step_revenue - degradation + timing_bonus + end_bonus
        reward = total * self.reward_scale

        # I validate
        if not np.isfinite(reward):
            reward = 0.0

        components['total_raw'] = total
        return reward, components


class AFRRRewardCalculator:
    """
    I calculate rewards for the aFRR (automatic Frequency Restoration Reserve) model.

    Revenue comes from:
    1. Capacity payments (if selected in merit order)
    2. Energy payments (if activated by TSO)

    Costs:
    1. Degradation from activation energy
    2. Non-response penalty (if activated but can't deliver)
    3. Opportunity cost (reserved capacity can't trade IntraDay)
    """

    def __init__(
        self,
        degradation_cost_per_mwh: float = 15.0,
        nonresponse_penalty: float = 2000.0,
        opportunity_cost_weight: float = 0.3,
        reward_scale: float = 0.001,
    ):
        self.degradation_cost = degradation_cost_per_mwh
        self.nonresponse_penalty = nonresponse_penalty
        self.opportunity_cost_weight = opportunity_cost_weight
        self.reward_scale = reward_scale

    def calculate(
        self,
        commitment_mw: float,
        cap_clearing_price: float,
        was_selected: bool,
        was_activated: bool,
        activation_energy_mwh: float,
        energy_price: float,
        shortfall_mw: float,
        capacity_mwh: float,
        lagged_id_spread: float = 4.0,
        block_hours: float = 4.0,
    ) -> Tuple[float, Dict]:
        """
        I calculate the reward for an aFRR block decision.

        Args:
            commitment_mw: Capacity committed in MW
            cap_clearing_price: Capacity clearing price (EUR/MW/h)
            was_selected: Whether capacity bid was selected
            was_activated: Whether TSO activated the reserve
            activation_energy_mwh: Energy delivered during activation
            energy_price: aFRR energy clearing price (EUR/MWh)
            shortfall_mw: MW shortfall if couldn't deliver when activated
            capacity_mwh: Battery capacity in MWh
            lagged_id_spread: Lagged IntraDay bid-ask spread
            block_hours: Block duration in hours

        Returns:
            (reward, info_dict) tuple
        """
        components = {}

        # I calculate capacity payment (earned if selected)
        cap_revenue = 0.0
        if was_selected:
            cap_revenue = commitment_mw * cap_clearing_price * block_hours
        components['capacity_revenue'] = cap_revenue

        # I calculate energy revenue (earned if activated)
        energy_revenue = 0.0
        if was_activated:
            energy_revenue = activation_energy_mwh * energy_price
        components['energy_revenue'] = energy_revenue

        # I calculate degradation from activation
        degradation = 0.0
        if was_activated:
            degradation = abs(activation_energy_mwh) * self.degradation_cost / capacity_mwh
        components['degradation'] = degradation

        # I penalize non-response (if activated but couldn't deliver full commitment)
        nonresponse_cost = 0.0
        if was_activated and shortfall_mw > 0.1:
            nonresponse_cost = shortfall_mw * self.nonresponse_penalty
        components['nonresponse'] = nonresponse_cost

        # I calculate opportunity cost (reserved capacity can't trade IntraDay)
        opportunity = commitment_mw * lagged_id_spread * self.opportunity_cost_weight
        components['opportunity_cost'] = opportunity

        total = cap_revenue + energy_revenue - degradation - nonresponse_cost - opportunity
        reward = total * self.reward_scale

        if not np.isfinite(reward):
            reward = 0.0

        components['total_raw'] = total
        return reward, components


class IntraDayRewardCalculator:
    """
    I calculate rewards for the IntraDay XBID model.

    Revenue comes from buying low / selling high on continuous IntraDay market.
    The agent trades at XBID bid (sell) and ask (buy) prices.
    """

    def __init__(
        self,
        degradation_cost_per_mwh: float = 15.0,
        spread_bonus_weight: float = 0.05,
        reward_scale: float = 0.001,
        soc_risk_penalty: float = 20.0,
    ):
        self.degradation_cost = degradation_cost_per_mwh
        self.spread_bonus_weight = spread_bonus_weight
        self.reward_scale = reward_scale
        self.soc_risk_penalty = soc_risk_penalty

    def calculate(
        self,
        action_mw: float,
        xbid_bid_price: float,
        xbid_ask_price: float,
        dam_price: float,
        capacity_mwh: float,
        time_step_hours: float = 0.25,
        soc: float = 0.5,
        shortfall_risk: float = 0.0,
    ) -> Tuple[float, Dict]:
        """
        I calculate the reward for an IntraDay XBID trade.

        Args:
            action_mw: Trade MW (+ = sell at bid, - = buy at ask)
            xbid_bid_price: XBID bid price (sell side)
            xbid_ask_price: XBID ask price (buy side)
            dam_price: DAM clearing price (for spread bonus)
            capacity_mwh: Battery capacity
            time_step_hours: ISP duration (0.25 for 15-min)
            soc: Current SoC
            shortfall_risk: Risk of not meeting future commitments [0,1]

        Returns:
            (reward, info_dict) tuple
        """
        components = {}
        energy_mwh = abs(action_mw) * time_step_hours

        # I calculate revenue based on trade direction
        revenue = 0.0
        if action_mw > 0.1:  # sell at bid
            revenue = energy_mwh * xbid_bid_price
        elif action_mw < -0.1:  # buy at ask (negative cost)
            revenue = -energy_mwh * xbid_ask_price
        components['revenue'] = revenue

        # I calculate degradation
        degradation = energy_mwh * self.degradation_cost / capacity_mwh
        components['degradation'] = degradation

        # I reward selling above DAM price (arbitrage)
        spread_bonus = 0.0
        if action_mw > 0.1 and xbid_bid_price > dam_price:
            spread_bonus = (xbid_bid_price - dam_price) * energy_mwh * self.spread_bonus_weight
        elif action_mw < -0.1 and xbid_ask_price < dam_price:
            spread_bonus = (dam_price - xbid_ask_price) * energy_mwh * self.spread_bonus_weight
        components['spread_bonus'] = spread_bonus

        # I penalize risky SoC positions
        risk_penalty = 0.0
        if shortfall_risk > 0.2:
            risk_penalty = self.soc_risk_penalty * (shortfall_risk - 0.2)
        components['risk_penalty'] = risk_penalty

        total = revenue - degradation + spread_bonus - risk_penalty
        reward = total * self.reward_scale

        if not np.isfinite(reward):
            reward = 0.0

        components['total_raw'] = total
        return reward, components


class MFRRRewardCalculator:
    """
    I calculate rewards for the mFRR (manual Frequency Restoration Reserve) model.

    Revenue comes from Free Bid activation (probabilistic, merit-order based).
    If not activated, there is no revenue and no cost.
    """

    def __init__(
        self,
        degradation_cost_per_mwh: float = 15.0,
        price_cap: float = 200.0,
        reward_scale: float = 0.001,
    ):
        self.degradation_cost = degradation_cost_per_mwh
        self.price_cap = price_cap
        self.reward_scale = reward_scale

    def calculate(
        self,
        bid_mw: float,
        bid_price: float,
        was_activated: bool,
        capacity_mwh: float,
        time_step_hours: float = 0.25,
    ) -> Tuple[float, Dict]:
        """
        I calculate the reward for an mFRR Free Bid.

        Args:
            bid_mw: Bid quantity in MW (absolute value)
            bid_price: Bid price in EUR/MWh
            was_activated: Whether the bid was activated by TSO
            capacity_mwh: Battery capacity
            time_step_hours: ISP duration (0.25 for 15-min)

        Returns:
            (reward, info_dict) tuple
        """
        components = {}
        energy_mwh = abs(bid_mw) * time_step_hours

        if was_activated:
            # I cap the settlement price
            settlement_price = min(bid_price, self.price_cap)
            revenue = energy_mwh * settlement_price
            degradation = energy_mwh * self.degradation_cost / capacity_mwh
        else:
            revenue = 0.0
            degradation = 0.0

        components['revenue'] = revenue
        components['degradation'] = degradation
        components['was_activated'] = float(was_activated)
        components['settlement_price'] = min(bid_price, self.price_cap) if was_activated else 0.0

        total = revenue - degradation
        reward = total * self.reward_scale

        if not np.isfinite(reward):
            reward = 0.0

        components['total_raw'] = total
        return reward, components
