"""
Reward Calculator with Market Depth & Imbalance Penalty.
Properly penalizes DAM non-compliance.

I apply SYMMETRIC penalties for ALL commitment violations:
1. SELL commitment (discharge) not honored → severe penalty
2. BUY commitment (charge) not honored → SAME severe penalty (NEW!)

This reflects real-world HEnEx rules: a commitment is a commitment,
regardless of direction. Both have financial AND administrative consequences.
"""
import numpy as np


class RewardCalculator:
    """
    Calculates reward based on proper market settlement.

    Key Fix: Not fulfilling DAM commitment = pay imbalance at worst price.

    CRITICAL: ANY failure to honor commitment incurs severe penalty:
    - SELL commitment not met: penalty (was already implemented)
    - BUY commitment not met: SAME penalty (NEW - symmetric treatment)

    This ensures the agent learns to respect ALL commitments equally.
    """
    def __init__(
        self,
        # TUNED v10: Encourage more trading, stricter DAM compliance
        deg_cost_per_mwh: float = 0.5,  # REDUCED from 2.0 - allow more cycling
        violation_penalty: float = 5.0,
        imbalance_penalty_multiplier: float = 20.0,
        fixed_shortfall_penalty: float = 800.0,     # INCREASED from 600 - stricter DAM
        proactive_risk_penalty: float = 50.0,       # REDUCED from 100 - less conservative
        proactive_risk_threshold: float = 0.20,     # RELAXED from 0.15
        soc_buffer_bonus: float = 2.0,              # REDUCED from 5.0 - less idle incentive
        min_soc_threshold: float = 0.25,            # RELAXED from 0.30
        # NEW: Price timing bonus to encourage buy-low-sell-high
        price_timing_bonus: float = 0.1,            # Bonus multiplier for good timing
        # v13: Lookahead timing rewards
        peak_sell_bonus: float = 10.0,              # Bonus for selling at/near peak
        early_sell_penalty: float = 15.0,           # Penalty for selling when better prices coming
        early_sell_threshold: float = 1.3,          # Penalty if max_future > current * threshold
        # v16: Price quartile awareness - penalize bad price decisions (ADDITIVE - deprecated)
        quartile_sell_cheap_penalty: float = 15.0,  # Penalty for selling in Q1 (cheap)
        quartile_buy_expensive_penalty: float = 15.0,  # Penalty for buying in Q4 (expensive)
        quartile_good_trade_bonus: float = 8.0,      # Bonus for good quartile trades
        # v18: MULTIPLICATIVE approach - scales with profit magnitude
        use_multiplicative_quartile: bool = False,  # Enable v18 multiplicative mode
        bad_trade_profit_multiplier: float = 0.3,   # Reduce profit by 70% for bad trades
        good_trade_profit_multiplier: float = 1.5,  # Boost profit by 50% for good trades
        # v19: Direct charge cost manipulation - affects actual balancing revenue
        use_charge_cost_scaling: bool = False,      # Enable v19 charge cost scaling
        expensive_charge_multiplier: float = 1.5,   # Multiply charge cost when price > 70th percentile
        cheap_charge_multiplier: float = 0.7,       # Multiply charge cost when price < 30th percentile
        charge_expensive_threshold: float = 0.7,    # Price percentile above which charging is "expensive"
        charge_cheap_threshold: float = 0.3         # Price percentile below which charging is "cheap"
    ):
        self.deg_cost_per_mwh = deg_cost_per_mwh
        self.violation_penalty = violation_penalty
        self.imbalance_penalty_multiplier = imbalance_penalty_multiplier
        self.fixed_shortfall_penalty = fixed_shortfall_penalty
        self.proactive_risk_penalty = proactive_risk_penalty
        self.proactive_risk_threshold = proactive_risk_threshold
        self.soc_buffer_bonus = soc_buffer_bonus
        self.min_soc_threshold = min_soc_threshold
        self.price_timing_bonus = price_timing_bonus
        # v13 timing
        self.peak_sell_bonus = peak_sell_bonus
        self.early_sell_penalty = early_sell_penalty
        self.early_sell_threshold = early_sell_threshold
        # v16 price quartile awareness (additive)
        self.quartile_sell_cheap_penalty = quartile_sell_cheap_penalty
        self.quartile_buy_expensive_penalty = quartile_buy_expensive_penalty
        self.quartile_good_trade_bonus = quartile_good_trade_bonus
        # v18 multiplicative approach
        self.use_multiplicative_quartile = use_multiplicative_quartile
        self.bad_trade_profit_multiplier = bad_trade_profit_multiplier
        self.good_trade_profit_multiplier = good_trade_profit_multiplier
        # v19 direct charge cost scaling
        self.use_charge_cost_scaling = use_charge_cost_scaling
        self.expensive_charge_multiplier = expensive_charge_multiplier
        self.cheap_charge_multiplier = cheap_charge_multiplier
        self.charge_expensive_threshold = charge_expensive_threshold
        self.charge_cheap_threshold = charge_cheap_threshold

    def calculate(
        self,
        dam_commitment: float,
        actual_physical: float,
        market_state,
        is_violation: bool = False,
        shortfall_risk: float = 0.0,  # I added this for proactive penalty
        current_soc: float = 0.5,     # I added this for SoC buffer bonus
        price_mean_24h: float = None, # I added this for price timing bonus (v10)
        # v13: Lookahead info for timing rewards
        max_future_price: float = None,  # Max price in next 12 hours
        has_dam_now: bool = False,       # Whether current hour has DAM commitment
        # v16: Price range for quartile calculation
        price_min_24h: float = None,
        price_max_24h: float = None
    ) -> dict:
        """
        Calculate P&L with proper imbalance settlement.

        I added proactive_risk_penalty: a small penalty applied when the agent
        is in a risky SoC state (shortfall_risk > 0), even before actual shortfall.
        This teaches the agent to maintain a safety buffer.

        Logic:
            - DAM commitment creates an OBLIGATION
            - If you don't fulfill it, you pay/receive at imbalance price

        Example 1: DAM commit = -10 (buy), actual = 0 (IDLE)
            → You didn't buy, so you SOLD your position to grid
            → imbalance = 0 - (-10) = +10
            → You get paid bid price (lower!) for the 10MW
            → Loss = dam_price - bid = spread/2

        Example 2: DAM commit = +10 (sell), actual = 0 (IDLE)
            → You didn't sell, so you BOUGHT from grid to cover
            → imbalance = 0 - 10 = -10
            → You pay ask price (higher!) for the 10MW
            → Loss = ask - dam_price = spread/2
        """
        
        # 1. DAM Revenue/Cost (This is LOCKED from day-ahead)
        # If we committed to buy (-), we pay dam_price
        # If we committed to sell (+), we receive dam_price
        dam_revenue = dam_commitment * market_state.dam_price

        # 2. Imbalance Settlement (THE FIX!)
        # Imbalance = actual - commitment
        # Positive imbalance (sold more / bought less): get BID price
        # Negative imbalance (sold less / bought more): pay ASK price
        imbalance_mw = actual_physical - dam_commitment

        # I check for "failure to honor commitment" scenarios:
        # This applies SYMMETRICALLY to both SELL and BUY commitments!
        # Any deviation from commitment has administrative + financial consequences.
        shortfall_mw = 0.0
        shortfall_penalty = 0.0

        if dam_commitment > 0 and actual_physical < dam_commitment:
            # CASE 1: FAILURE TO DELIVER ON SELL
            # Promised to sell X MW, delivered less (or even charged!)
            # Example: commitment +15MW, actual +8MW → shortfall 7MW
            # Example: commitment +15MW, actual -5MW → shortfall 20MW (severe!)
            shortfall_mw = dam_commitment - actual_physical

            shortfall_penalty = shortfall_mw * (
                market_state.best_ask * self.imbalance_penalty_multiplier
                + self.fixed_shortfall_penalty
            )

        elif dam_commitment < 0 and actual_physical > dam_commitment:
            # CASE 2: FAILURE TO HONOR BUY COMMITMENT (NEW!)
            # Promised to buy X MW, bought less (or even discharged!)
            # Example: commitment -10MW, actual -5MW → shortfall 5MW
            # Example: commitment -10MW, actual +5MW → shortfall 15MW (severe!)
            shortfall_mw = actual_physical - dam_commitment  # Both negative, so this gives positive

            # I apply the SAME penalty - a commitment is a commitment!
            shortfall_penalty = shortfall_mw * (
                market_state.best_ask * self.imbalance_penalty_multiplier
                + self.fixed_shortfall_penalty
            )

        if imbalance_mw > 0:
            # We delivered MORE than committed (or bought LESS)
            # Grid pays us at BID price (lower than mid)
            balancing_revenue = imbalance_mw * market_state.best_bid
        elif imbalance_mw < 0:
            # We delivered LESS than committed (or bought MORE)
            # We pay grid at ASK price (higher than mid)
            balancing_revenue = imbalance_mw * market_state.best_ask
        else:
            # Perfect execution
            balancing_revenue = 0.0

        # v19: Direct charge cost scaling
        # I manipulate the ACTUAL cost of charging based on price percentile
        # This affects net_profit directly, not as a separate penalty
        charge_cost_multiplier = 1.0
        if self.use_charge_cost_scaling and actual_physical < -0.5:  # Charging
            # Calculate price percentile if we have the data
            if price_min_24h is not None and price_max_24h is not None:
                current_price = market_state.dam_price
                price_range = price_max_24h - price_min_24h
                if price_range > 1.0:
                    price_pct = (current_price - price_min_24h) / price_range
                    price_pct = max(0.0, min(1.0, price_pct))

                    # Only apply to IntraDay (no DAM commitment)
                    if not has_dam_now:
                        if price_pct > self.charge_expensive_threshold:
                            # Charging at expensive prices - increase perceived cost
                            charge_cost_multiplier = self.expensive_charge_multiplier
                        elif price_pct < self.charge_cheap_threshold:
                            # Charging at cheap prices - decrease perceived cost
                            charge_cost_multiplier = self.cheap_charge_multiplier

            # Apply multiplier to balancing_revenue (which is negative for charging)
            # Multiplying negative by >1 makes it more negative (higher cost)
            # Multiplying negative by <1 makes it less negative (lower cost)
            balancing_revenue = balancing_revenue * charge_cost_multiplier

        # 3. Degradation Cost (only for ACTUAL physical activity)
        degradation = abs(actual_physical) * self.deg_cost_per_mwh

        # 4. Net Financial P&L (now includes shortfall penalty!)
        net_profit = dam_revenue + balancing_revenue - degradation - shortfall_penalty

        # 5. Spread Loss Metric
        spread = market_state.best_ask - market_state.best_bid
        spread_loss = abs(imbalance_mw) * spread / 2

        # 6. Safety Penalty (physical violations only)
        penalty = -self.violation_penalty if is_violation else 0.0

        # 7. Proactive Risk Penalty (I apply this when SoC is risky, BEFORE shortfall)
        # This teaches the agent to maintain a safety buffer
        # I lowered threshold and made penalty scale more aggressively
        proactive_penalty = 0.0
        if shortfall_risk > self.proactive_risk_threshold:
            # Scale penalty with risk level, more aggressive scaling
            risk_factor = (shortfall_risk - self.proactive_risk_threshold) / (1.0 - self.proactive_risk_threshold)
            # Quadratic scaling - small risk = small penalty, high risk = BIG penalty
            proactive_penalty = self.proactive_risk_penalty * (risk_factor ** 1.5)

        # 8. SoC Buffer Bonus (I reward maintaining healthy SoC levels)
        # This encourages the agent to keep a safety buffer
        soc_bonus = 0.0
        if current_soc >= self.min_soc_threshold:
            # Bonus scales with how much above threshold
            buffer_factor = (current_soc - self.min_soc_threshold) / (1.0 - self.min_soc_threshold)
            soc_bonus = self.soc_buffer_bonus * buffer_factor
        else:
            # Penalty for being below threshold (symmetric with bonus)
            deficit_factor = (self.min_soc_threshold - current_soc) / self.min_soc_threshold
            soc_bonus = -self.soc_buffer_bonus * deficit_factor * 2  # Double penalty for low SoC

        # 9. Price Timing Bonus (v10) - I reward buy-low-sell-high behavior
        # This teaches the agent to trade at favorable prices
        timing_bonus = 0.0
        if price_mean_24h is not None and price_mean_24h > 0:
            current_price = market_state.dam_price
            price_deviation = (current_price - price_mean_24h) / price_mean_24h  # Normalized deviation

            if actual_physical > 0.1:  # Discharging (selling)
                # I reward selling when price is ABOVE average
                timing_bonus = abs(actual_physical) * price_deviation * self.price_timing_bonus * 100
            elif actual_physical < -0.1:  # Charging (buying)
                # I reward buying when price is BELOW average (negative deviation = bonus)
                timing_bonus = abs(actual_physical) * (-price_deviation) * self.price_timing_bonus * 100

        # 10. v13 Lookahead Timing Reward/Penalty
        # I teach the agent to use the 12-hour lookahead for optimal timing
        lookahead_timing = 0.0
        if max_future_price is not None and actual_physical > 0.1:  # Only for selling
            current_price = market_state.dam_price
            price_ratio = max_future_price / max(current_price, 1.0)

            # Case A: Selling at/near peak - BONUS!
            if current_price >= max_future_price * 0.95:
                # Great timing! Selling at the best price
                lookahead_timing = self.peak_sell_bonus * (actual_physical / 30.0)

            # Case B: Selling with high upside and NO DAM forcing it - PENALTY!
            elif price_ratio > self.early_sell_threshold and not has_dam_now:
                # Bad timing! Could have waited for better price
                # Penalty scales with how much upside was missed
                upside_missed = (price_ratio - 1.0)  # e.g., 0.5 for 50% upside
                lookahead_timing = -self.early_sell_penalty * upside_missed * (actual_physical / 30.0)

        # 11. v16/v18: Price Quartile Awareness
        # I penalize selling in Q1 (cheap) and buying in Q4 (expensive)
        # This teaches the agent to respect basic buy-low-sell-high logic
        quartile_adjustment = 0.0
        price_percentile = 0.5  # Default to middle
        profit_multiplier = 1.0  # v18: multiplicative approach

        if price_min_24h is not None and price_max_24h is not None:
            current_price = market_state.dam_price
            price_range = price_max_24h - price_min_24h

            if price_range > 1.0:  # Avoid division by zero
                price_percentile = (current_price - price_min_24h) / price_range
                price_percentile = max(0.0, min(1.0, price_percentile))  # Clamp to [0, 1]

                # Only apply to IntraDay trades (no DAM commitment)
                if not has_dam_now:
                    is_bad_trade = False
                    is_good_trade = False

                    if actual_physical > 0.5:  # Discharging (selling)
                        if price_percentile < 0.25:
                            is_bad_trade = True  # Selling cheap
                        elif price_percentile > 0.75:
                            is_good_trade = True  # Selling expensive

                    elif actual_physical < -0.5:  # Charging (buying)
                        if price_percentile > 0.75:
                            is_bad_trade = True  # Buying expensive
                        elif price_percentile < 0.25:
                            is_good_trade = True  # Buying cheap

                    # v18: Apply MULTIPLICATIVE penalty/bonus
                    if self.use_multiplicative_quartile:
                        if is_bad_trade:
                            # I reduce the profit significantly for bad trades
                            profit_multiplier = self.bad_trade_profit_multiplier
                        elif is_good_trade:
                            # I boost the profit for good trades
                            profit_multiplier = self.good_trade_profit_multiplier
                    else:
                        # v16: Original ADDITIVE approach
                        if actual_physical > 0.5:  # Selling
                            if price_percentile < 0.25:
                                quartile_adjustment = -self.quartile_sell_cheap_penalty * (0.25 - price_percentile) * 4
                            elif price_percentile > 0.75:
                                quartile_adjustment = self.quartile_good_trade_bonus * (price_percentile - 0.75) * 4
                        elif actual_physical < -0.5:  # Buying
                            if price_percentile > 0.75:
                                quartile_adjustment = -self.quartile_buy_expensive_penalty * (price_percentile - 0.75) * 4
                            elif price_percentile < 0.25:
                                quartile_adjustment = self.quartile_good_trade_bonus * (0.25 - price_percentile) * 4

        # 12. Total Reward
        # v18: Apply multiplicative factor to net_profit before adding other components
        adjusted_profit = net_profit * profit_multiplier
        total_reward = adjusted_profit + penalty - proactive_penalty + soc_bonus + timing_bonus + lookahead_timing + quartile_adjustment

        return {
            "reward": total_reward / 100.0,  # Scale for NN
            "components": {
                "dam_rev": dam_revenue,
                "balancing_rev": balancing_revenue,
                "degradation": degradation,
                "shortfall_mw": shortfall_mw,
                "shortfall_penalty": shortfall_penalty,
                "proactive_penalty": proactive_penalty,
                "soc_bonus": soc_bonus,
                "timing_bonus": timing_bonus,  # v10
                "lookahead_timing": lookahead_timing,  # v13 - peak/early sell reward
                "quartile_adj": quartile_adjustment,  # v16 - price quartile penalty/bonus
                "price_pct": price_percentile,  # v16 - for debugging
                "profit_mult": profit_multiplier,  # v18 - multiplicative factor applied
                "charge_cost_mult": charge_cost_multiplier,  # v19 - charge cost scaling factor
                "net_profit": net_profit,  # Original profit before multiplier
                "adjusted_profit": adjusted_profit,  # v18 - profit after multiplier
                "spread": spread,
                "spread_loss": spread_loss,
                "imbalance_mw": imbalance_mw,
                "penalty": penalty
            }
        }
