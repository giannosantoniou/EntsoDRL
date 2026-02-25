"""
Unified Multi-Market Reward Calculator

I calculate rewards for the unified multi-market environment that handles:
- DAM commitment compliance
- IntraDay energy trading
- aFRR capacity payments and energy activation
- mFRR energy trading

The reward formula ensures proper incentives for all market activities while
maintaining strict compliance with HEnEx regulatory requirements.

Total Reward = DAM_Revenue + IntraDay_Revenue + aFRR_Capacity + aFRR_Energy + mFRR_Revenue
               - Degradation - Calendar_Aging - DAM_Violation_Penalty - aFRR_NonResponse_Penalty
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class UnifiedMarketState:
    """I hold all market prices and states for reward calculation."""
    # DAM
    dam_price: float
    dam_commitment: float  # MW (positive = sell, negative = buy)

    # IntraDay
    intraday_bid: float  # Best bid price
    intraday_ask: float  # Best ask price
    intraday_spread: float

    # aFRR
    afrr_cap_up_price: float  # EUR/MW/h for upward capacity
    afrr_cap_down_price: float  # EUR/MW/h for downward capacity
    afrr_energy_up_price: float  # EUR/MWh when activated up
    afrr_energy_down_price: float  # EUR/MWh when activated down
    afrr_activated: bool
    afrr_activation_direction: Optional[str]  # 'up' or 'down'

    # mFRR
    mfrr_price_up: float  # EUR/MWh for upward energy
    mfrr_price_down: float  # EUR/MWh for downward energy

    # IDA sub-market prices (full market mode)
    ida1_clearing_price: float = 0.0
    ida2_clearing_price: float = 0.0
    ida3_clearing_price: float = 0.0
    net_ida_position: float = 0.0      # MW locked position

    # XBID prices (replaces intraday for full market mode)
    xbid_bid: float = 0.0
    xbid_ask: float = 0.0

    # Free Bid (agent-set price, pay-as-bid)
    free_bid_activated: bool = False
    free_bid_agent_price: float = 0.0  # The price the AGENT set


class UnifiedRewardCalculator:
    """
    I calculate rewards for the unified multi-market battery environment.

    Key Design Decisions:
    1. DAM violations checked against DAM-ONLY energy (not net total) to
       avoid false penalties when IntraDay/aFRR trade in opposite direction
    2. aFRR non-response penalty = 500 EUR/MWh (reduced from 2000)
    3. Degradation = 5 EUR/MWh (reduced from 15 — realistic for Li-ion BESS
       and allows IntraDay profitability with mean spread ~11 EUR/MWh)
    4. aFRR capacity payments are unconditional when selected
    5. All revenues scale with actual delivered energy
    6. reward_scale = 0.01 (raised from 0.001 for stronger NN signal)
    7. dam_balancing term removed (was double-counting non-DAM revenue)
    """

    def __init__(
        self,
        # Degradation
        degradation_cost_per_mwh: float = 5.0,  # I reduced from 15 to 5 EUR/MWh
        # (realistic Li-ion BESS cost, enables IntraDay with mean spread ~11 EUR/MWh)

        # Penalty coefficients — I reduced these to fix the 37:1 penalty/profit
        # asymmetry that was destroying the gradient signal. The old values
        # (1500/2000) caused single-step penalties of 11-15k EUR while typical
        # profits were ~300 EUR, making SNR ≈ 0.1 (need >0.5 for stable PPO).
        dam_violation_penalty: float = 800.0,  # EUR/MWh - was 1500, still deterrent
        afrr_nonresponse_penalty: float = 500.0,  # EUR/MWh - was 2000, still severe
        physical_violation_penalty: float = 100.0,  # EUR for physically impossible actions

        # Bonuses
        dam_compliance_bonus: float = 5.0,  # EUR/MWh for perfect DAM compliance
        afrr_response_bonus: float = 10.0,  # EUR/MWh for fast aFRR response

        # Cycle management — I penalize daily cycling above target to keep
        # battery wear at 1.5-2.0 cycles/day. Super-linear scaling means
        # small overages get mild penalty, large overages get severe penalty.
        cycle_target: float = 2.0,  # Target daily cycles
        cycle_excess_penalty: float = 3000.0,  # EUR per excess cycle (super-linear)

        # Calendar aging — Li-ion batteries degrade faster at extreme SoC.
        # I model this as a quadratic cost: (SoC - 0.5)^2 * coefficient per step.
        # At 50% SoC: 0 EUR. At 5% SoC: 0.2025 * 200 = 40.5 EUR/step.
        # At 20% SoC: 0.09 * 200 = 18 EUR/step (~30% of trading profit).
        # I raised from 50 to 200 because the agent was spending 45% of time
        # below 20% SoC — the old penalty was only 7.5% of per-step profit.
        calendar_aging_coefficient: float = 200.0,

        # Scaling — I raised from 0.001 to 0.01 so that a typical 300 EUR
        # trade produces a reward of 3.0 instead of 0.3, giving the NN a
        # much stronger learning signal without needing VecNormalize reward norm.
        reward_scale: float = 0.01  # Scale rewards for NN training (was 0.001)
    ):
        self.degradation_cost = degradation_cost_per_mwh
        self.dam_violation_penalty = dam_violation_penalty
        self.afrr_nonresponse_penalty = afrr_nonresponse_penalty
        self.physical_violation_penalty = physical_violation_penalty
        self.dam_compliance_bonus = dam_compliance_bonus
        self.afrr_response_bonus = afrr_response_bonus
        self.cycle_target = cycle_target
        self.cycle_excess_penalty = cycle_excess_penalty
        self.calendar_aging_coefficient = calendar_aging_coefficient
        self.reward_scale = reward_scale

    def calculate(
        self,
        market: UnifiedMarketState,
        # Actions taken
        actual_energy_mw: float,  # Total net energy (DAM+aFRR+ID+mFRR)
        afrr_capacity_committed_mw: float,  # aFRR capacity commitment
        afrr_energy_delivered_mw: float,  # aFRR energy when activated

        # Battery state
        current_soc: float,
        capacity_mwh: float,

        # Optional actions (default 0 for backward compatibility)
        intraday_energy_mw: float = 0.0,  # IntraDay energy traded (positive = sell)
        mfrr_energy_mw: float = 0.0,  # mFRR energy traded
        time_step_hours: float = 1.0,

        # Context
        is_physical_violation: bool = False,
        shortfall_risk: float = 0.0,
        price_momentum: float = 0.0,  # Backward-looking price trend
        is_selected_for_afrr: bool = True,

        # Full market mode (IDA/XBID/FreeBid)
        ida_energy_mw: float = 0.0,
        xbid_energy_mw: float = 0.0,
        free_bid_energy_mw: float = 0.0,

        # I added dam_executed_mw so DAM compliance is checked against
        # DAM-only energy, not the net total. Before this fix, IntraDay
        # or aFRR trades in the opposite direction of DAM triggered
        # false 800 EUR/MWh violations.
        dam_executed_mw: float = None,

        # I added afrr_max_deliverable_mw to distinguish between agent fault
        # (chose wrong action) and SoC starvation (DAM consumed the energy).
        # When set and < 0.1, I skip the non-response penalty because the
        # agent physically couldn't respond — not its fault.
        afrr_max_deliverable_mw: float = None,

        # I pass daily_cycles so the reward calculator can apply super-linear
        # penalty when cycling exceeds the target (1.5-2.0/day).
        daily_cycles: float = 0.0,
    ) -> Dict:
        """
        I calculate the total reward for a unified multi-market step.

        Returns a dict with:
        - reward: scaled total reward
        - components: detailed breakdown of each component
        """
        components = {}

        # I use dam_executed_mw for compliance checks (not net total).
        # If caller doesn't provide it, I fall back to actual_energy_mw
        # for backward compatibility.
        dam_delivery_mw = dam_executed_mw if dam_executed_mw is not None else actual_energy_mw

        # =====================================================================
        # 1. DAM SETTLEMENT
        # =====================================================================
        # I calculate DAM revenue/cost based on commitment
        dam_commitment = market.dam_commitment
        dam_revenue = dam_commitment * market.dam_price * time_step_hours
        components['dam_revenue'] = dam_revenue

        # I removed the old dam_balancing term because it double-counted
        # revenue from IntraDay/mFRR/aFRR. The net-flow imbalance was
        # priced at bid/ask AND each market's revenue was added separately,
        # inflating total revenue by 2x for non-DAM trades.
        components['dam_balancing'] = 0.0

        # I check DAM compliance against DAM-ONLY delivered energy, not the
        # net total. Before this fix, IntraDay buy + DAM sell = false violation.
        dam_shortfall_mw = 0.0
        dam_violation_cost = 0.0

        if dam_commitment > 0 and dam_delivery_mw < dam_commitment - 0.1:
            # I failed to deliver on SELL commitment
            dam_shortfall_mw = dam_commitment - dam_delivery_mw
            dam_violation_cost = dam_shortfall_mw * self.dam_violation_penalty * time_step_hours
        elif dam_commitment < 0 and dam_delivery_mw > dam_commitment + 0.1:
            # I failed to honor BUY commitment (dam_delivery_mw should be negative)
            dam_shortfall_mw = abs(dam_delivery_mw - dam_commitment)
            dam_violation_cost = dam_shortfall_mw * self.dam_violation_penalty * time_step_hours

        components['dam_shortfall_mw'] = dam_shortfall_mw
        components['dam_violation_cost'] = dam_violation_cost

        # I check DAM compliance bonus against DAM-only energy
        dam_bonus = 0.0
        if abs(dam_commitment) > 0.1 and dam_shortfall_mw < 0.1:
            dam_bonus = abs(dam_commitment) * self.dam_compliance_bonus * time_step_hours
        components['dam_bonus'] = dam_bonus

        # =====================================================================
        # 2. aFRR CAPACITY REVENUE
        # =====================================================================
        # I calculate aFRR capacity payment (paid for availability)
        afrr_capacity_revenue = 0.0
        if is_selected_for_afrr and afrr_capacity_committed_mw > 0.1:
            # I get paid for committed capacity
            afrr_capacity_revenue = afrr_capacity_committed_mw * market.afrr_cap_up_price * time_step_hours
        components['afrr_capacity_revenue'] = afrr_capacity_revenue

        # =====================================================================
        # 3. aFRR ENERGY REVENUE (when activated)
        # =====================================================================
        afrr_energy_revenue = 0.0
        afrr_nonresponse_cost = 0.0

        if market.afrr_activated and is_selected_for_afrr and afrr_capacity_committed_mw > 0.1:
            # TSO activated aFRR - I MUST respond
            required_power = afrr_capacity_committed_mw

            if market.afrr_activation_direction == 'up':
                # Must discharge
                if afrr_energy_delivered_mw > 0.1:
                    # Good - I responded
                    afrr_energy_revenue = afrr_energy_delivered_mw * market.afrr_energy_up_price * time_step_hours
                    # I add response bonus for timely compliance
                    if afrr_energy_delivered_mw >= required_power * 0.9:
                        afrr_energy_revenue += required_power * self.afrr_response_bonus * time_step_hours
                else:
                    # I check if the agent was SoC-starved (DAM consumed the
                    # energy). If so, I skip the penalty — not the agent's fault.
                    if afrr_max_deliverable_mw is not None and afrr_max_deliverable_mw < 0.1:
                        afrr_nonresponse_cost = 0.0
                    else:
                        # BAD - I failed to respond (SEVERE penalty)
                        afrr_nonresponse_cost = required_power * self.afrr_nonresponse_penalty * time_step_hours

            elif market.afrr_activation_direction == 'down':
                # Must charge
                if afrr_energy_delivered_mw < -0.1:
                    # Good - I responded (delivered is negative for charging)
                    afrr_energy_revenue = abs(afrr_energy_delivered_mw) * market.afrr_energy_down_price * time_step_hours
                    if abs(afrr_energy_delivered_mw) >= required_power * 0.9:
                        afrr_energy_revenue += required_power * self.afrr_response_bonus * time_step_hours
                else:
                    # I check if the agent was SoC-starved (no headroom to charge).
                    if afrr_max_deliverable_mw is not None and afrr_max_deliverable_mw < 0.1:
                        afrr_nonresponse_cost = 0.0
                    else:
                        # BAD - I failed to respond
                        afrr_nonresponse_cost = required_power * self.afrr_nonresponse_penalty * time_step_hours

        components['afrr_energy_revenue'] = afrr_energy_revenue
        components['afrr_nonresponse_cost'] = afrr_nonresponse_cost

        # =====================================================================
        # 4. INTRADAY ENERGY REVENUE
        # =====================================================================
        intraday_revenue = 0.0
        if abs(intraday_energy_mw) > 0.1:
            if intraday_energy_mw > 0:  # Sell at bid
                intraday_revenue = intraday_energy_mw * market.intraday_bid * time_step_hours
            else:  # Buy at ask
                intraday_revenue = intraday_energy_mw * market.intraday_ask * time_step_hours
        components['intraday_revenue'] = intraday_revenue

        # =====================================================================
        # 5. mFRR ENERGY REVENUE
        # =====================================================================
        mfrr_revenue = 0.0
        if abs(mfrr_energy_mw) > 0.1:
            if mfrr_energy_mw > 0:
                # Discharging to mFRR (selling upward regulation)
                mfrr_revenue = mfrr_energy_mw * market.mfrr_price_up * time_step_hours
            else:
                # Charging from mFRR (providing downward regulation)
                # I use abs() because down-regulation is a SERVICE — the TSO
                # pays the BSP to absorb excess energy. Revenue is positive.
                mfrr_revenue = abs(mfrr_energy_mw) * market.mfrr_price_down * time_step_hours

        components['mfrr_revenue'] = mfrr_revenue

        # =====================================================================
        # 5b. IDA SETTLEMENT (pay-as-cleared, locked positions)
        # =====================================================================
        ida_revenue = 0.0
        if abs(ida_energy_mw) > 0.1:
            # I use weighted average of IDA clearing prices
            ida_revenue = ida_energy_mw * self._weighted_ida_price(market) * time_step_hours
        components['ida_revenue'] = ida_revenue

        # =====================================================================
        # 5c. XBID SETTLEMENT (pay-as-bid, continuous)
        # =====================================================================
        xbid_revenue = 0.0
        if abs(xbid_energy_mw) > 0.1:
            if xbid_energy_mw > 0:
                # I sell at XBID bid price
                xbid_revenue = xbid_energy_mw * market.xbid_bid * time_step_hours
            else:
                # I buy at XBID ask price
                xbid_revenue = xbid_energy_mw * market.xbid_ask * time_step_hours
        components['xbid_revenue'] = xbid_revenue

        # =====================================================================
        # 5d. FREE BID SETTLEMENT (pay-as-bid, agent's price)
        # =====================================================================
        free_bid_revenue = 0.0
        if market.free_bid_activated and abs(free_bid_energy_mw) > 0.1:
            # I use the agent's bid price (pay-as-bid settlement)
            free_bid_revenue = abs(free_bid_energy_mw) * market.free_bid_agent_price * time_step_hours
            if free_bid_energy_mw < 0:
                free_bid_revenue = -free_bid_revenue
        components['free_bid_revenue'] = free_bid_revenue

        # =====================================================================
        # 6. DEGRADATION COST
        # =====================================================================
        # I calculate degradation based on total energy cycled
        total_energy_cycled = abs(actual_energy_mw) * time_step_hours
        cycle_fraction = total_energy_cycled / capacity_mwh
        degradation_cost = cycle_fraction * capacity_mwh * self.degradation_cost

        components['degradation_cost'] = degradation_cost
        components['cycle_fraction'] = cycle_fraction

        # =====================================================================
        # 7. PHYSICAL VIOLATION PENALTY
        # =====================================================================
        physical_penalty = self.physical_violation_penalty if is_physical_violation else 0.0
        components['physical_penalty'] = physical_penalty

        # I removed sections 8 (risk_penalty), 9 (soc_bonus), 10 (timing_bonus)
        # because they added noise without meaningful signal. The action masking
        # already handles SoC safety, and the raw trade P&L is a cleaner signal
        # than small bonuses/penalties that were dwarfed by penalty spikes.
        # I still report them as zero for backward compatibility with loggers.
        components['risk_penalty'] = 0.0
        components['soc_bonus'] = 0.0
        components['timing_bonus'] = 0.0

        # =====================================================================
        # 7b. CYCLE EXCESS PENALTY (super-linear)
        # =====================================================================
        # I penalize daily cycling above target with super-linear scaling.
        # At 2.0 target: 2.5 cycles → penalty = 3000 × 0.5^1.5 = 1,060 EUR
        #                3.0 cycles → penalty = 3000 × 1.0^1.5 = 3,000 EUR
        #                5.0 cycles → penalty = 3000 × 3.0^1.5 = 15,588 EUR
        cycle_excess_cost = 0.0
        if daily_cycles > self.cycle_target:
            excess = daily_cycles - self.cycle_target
            cycle_excess_cost = self.cycle_excess_penalty * (excess ** 1.5)
        components['cycle_excess_cost'] = cycle_excess_cost

        # =====================================================================
        # 7c. CALENDAR AGING COST
        # =====================================================================
        # I model calendar aging as a quadratic penalty on SoC distance from 50%.
        # Li-ion batteries degrade faster at extreme SoC (high voltage stress at
        # top, copper dissolution at bottom). The cost is per time step.
        soc_deviation = current_soc - 0.5
        calendar_aging_cost = (soc_deviation ** 2) * self.calendar_aging_coefficient
        components['calendar_aging_cost'] = calendar_aging_cost

        # =====================================================================
        # 8. TOTAL REWARD
        # =====================================================================
        total_revenue = (
            dam_revenue + dam_bonus +
            afrr_capacity_revenue + afrr_energy_revenue +
            intraday_revenue +
            ida_revenue + xbid_revenue +
            free_bid_revenue + mfrr_revenue
        )

        total_costs = (
            degradation_cost +
            dam_violation_cost +
            afrr_nonresponse_cost +
            physical_penalty +
            cycle_excess_cost +
            calendar_aging_cost
        )

        net_profit = total_revenue - total_costs
        components['net_profit'] = net_profit

        # I scale the reward for NN training
        scaled_reward = net_profit * self.reward_scale

        # I validate the reward
        if not np.isfinite(scaled_reward):
            print(f"WARNING: Non-finite reward detected: {scaled_reward}")
            print(f"Components: {components}")
            scaled_reward = 0.0

        return {
            'reward': scaled_reward,
            'components': components
        }

    def _weighted_ida_price(self, market: UnifiedMarketState) -> float:
        """I compute a weighted average of IDA clearing prices for settlement."""
        prices = []
        if market.ida1_clearing_price > 0:
            prices.append(market.ida1_clearing_price)
        if market.ida2_clearing_price > 0:
            prices.append(market.ida2_clearing_price)
        if market.ida3_clearing_price > 0 and not np.isnan(market.ida3_clearing_price):
            prices.append(market.ida3_clearing_price)
        if len(prices) == 0:
            return market.dam_price  # I fall back to DAM price
        return np.mean(prices)

    def calculate_simple(
        self,
        dam_profit: float,
        afrr_capacity_profit: float,
        afrr_energy_profit: float,
        intraday_profit: float = 0.0,
        mfrr_profit: float = 0.0,
        degradation_cost: float = 0.0,
        penalties: float = 0.0
    ) -> float:
        """
        I provide a simplified reward calculation for quick evaluation.

        This is useful for backtesting and analysis where full market state
        is not available.
        """
        total = (
            dam_profit +
            afrr_capacity_profit +
            afrr_energy_profit +
            intraday_profit +
            mfrr_profit -
            degradation_cost -
            penalties
        )
        return total * self.reward_scale
