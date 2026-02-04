"""
Tests for RewardCalculator with Shortfall Penalty.

I verify that the "failure to deliver" penalty is applied correctly
when we commit to sell in DAM but can't physically deliver.
"""
import pytest
from dataclasses import dataclass


@dataclass
class MockMarketState:
    """Mock market state for testing."""
    dam_price: float
    best_bid: float
    best_ask: float
    spread: float


class TestRewardCalculator:
    """Test suite for RewardCalculator."""

    def setup_method(self):
        """I set up a fresh RewardCalculator for each test."""
        from gym_envs.reward_calculator import RewardCalculator
        self.calc = RewardCalculator(
            deg_cost_per_mwh=10.0,
            violation_penalty=5.0,
            imbalance_penalty_multiplier=2.0,
            fixed_shortfall_penalty=50.0
        )
        # Typical market: DAM=100, bid=98, ask=102
        self.market = MockMarketState(
            dam_price=100.0,
            best_bid=98.0,
            best_ask=102.0,
            spread=4.0
        )

    def test_perfect_execution_no_penalty(self):
        """I verify no shortfall penalty when we deliver exactly what we promised."""
        result = self.calc.calculate(
            dam_commitment=20.0,  # Promised to sell 20 MW
            actual_physical=20.0,  # Delivered 20 MW
            market_state=self.market
        )

        assert result['components']['shortfall_mw'] == 0.0
        assert result['components']['shortfall_penalty'] == 0.0
        assert result['components']['imbalance_mw'] == 0.0

    def test_over_delivery_no_penalty(self):
        """I verify no shortfall penalty when we deliver MORE than promised."""
        result = self.calc.calculate(
            dam_commitment=20.0,  # Promised to sell 20 MW
            actual_physical=25.0,  # Delivered 25 MW (5 extra)
            market_state=self.market
        )

        assert result['components']['shortfall_mw'] == 0.0
        assert result['components']['shortfall_penalty'] == 0.0
        assert result['components']['imbalance_mw'] == 5.0  # Positive imbalance

    def test_failure_to_deliver_severe_penalty(self):
        """
        CRITICAL TEST: I verify severe penalty when we promise to sell but can't deliver.

        Scenario: DAM commitment = +20 MW (sell), actual = +10 MW (only delivered half)
        Shortfall = 10 MW

        Penalty should be: 10 MW × (102€ × 2.0 + 50€) = 10 × 254 = 2540€
        """
        result = self.calc.calculate(
            dam_commitment=20.0,  # Promised to sell 20 MW
            actual_physical=10.0,  # Could only deliver 10 MW
            market_state=self.market
        )

        assert result['components']['shortfall_mw'] == 10.0
        # Expected: 10 × (102 × 2.0 + 50) = 10 × 254 = 2540
        expected_penalty = 10.0 * (102.0 * 2.0 + 50.0)
        assert result['components']['shortfall_penalty'] == expected_penalty
        assert result['components']['imbalance_mw'] == -10.0  # Negative imbalance

    def test_complete_failure_to_deliver(self):
        """
        I test the worst case: promised to sell but delivered NOTHING (SoC=0).

        Scenario: DAM commitment = +30 MW, actual = 0 MW (battery empty!)
        Shortfall = 30 MW

        Penalty should be: 30 MW × (102€ × 2.0 + 50€) = 30 × 254 = 7620€
        """
        result = self.calc.calculate(
            dam_commitment=30.0,  # Promised to sell 30 MW
            actual_physical=0.0,   # Couldn't deliver anything
            market_state=self.market
        )

        assert result['components']['shortfall_mw'] == 30.0
        expected_penalty = 30.0 * (102.0 * 2.0 + 50.0)
        assert result['components']['shortfall_penalty'] == expected_penalty

    def test_buying_no_shortfall_penalty(self):
        """I verify no shortfall penalty for buying scenarios (negative commitment)."""
        result = self.calc.calculate(
            dam_commitment=-20.0,  # Promised to BUY 20 MW
            actual_physical=-10.0,  # Only bought 10 MW
            market_state=self.market
        )

        # Shortfall penalty only applies to SELLING failures
        assert result['components']['shortfall_mw'] == 0.0
        assert result['components']['shortfall_penalty'] == 0.0

    def test_high_price_amplifies_penalty(self):
        """I verify penalty scales with market price during high-price periods."""
        high_price_market = MockMarketState(
            dam_price=300.0,
            best_bid=295.0,
            best_ask=305.0,
            spread=10.0
        )

        result = self.calc.calculate(
            dam_commitment=10.0,
            actual_physical=5.0,  # 5 MW shortfall
            market_state=high_price_market
        )

        # Shortfall = 5 MW × (305€ × 2.0 + 50€) = 5 × 660 = 3300€
        expected_penalty = 5.0 * (305.0 * 2.0 + 50.0)
        assert result['components']['shortfall_penalty'] == expected_penalty

    def test_reward_is_negative_for_large_shortfall(self):
        """I verify that large shortfalls result in negative total reward."""
        result = self.calc.calculate(
            dam_commitment=30.0,
            actual_physical=0.0,
            market_state=self.market
        )

        # DAM revenue: 30 × 100 = 3000€
        # Balancing: -30 × 102 = -3060€ (buy back at ask)
        # Degradation: 0 (no activity)
        # Shortfall penalty: 7620€
        # Net should be very negative
        assert result['reward'] < 0
        assert result['components']['net_profit'] < 0


class TestRewardCalculatorIntegration:
    """Integration tests for realistic scenarios."""

    def test_realistic_battery_cycle(self):
        """
        I simulate a realistic scenario where battery runs low mid-day.

        Morning: Committed to sell 25 MW for afternoon peak
        Reality: Only 15 MW worth of charge left
        Shortfall: 10 MW at 150€/MWh peak price
        """
        from gym_envs.reward_calculator import RewardCalculator
        calc = RewardCalculator()

        peak_market = MockMarketState(
            dam_price=150.0,
            best_bid=145.0,
            best_ask=155.0,
            spread=10.0
        )

        result = calc.calculate(
            dam_commitment=25.0,
            actual_physical=15.0,
            market_state=peak_market
        )

        # Shortfall = 10 MW × (155 × 2 + 50) = 10 × 360 = 3600€ penalty
        assert result['components']['shortfall_mw'] == 10.0
        assert result['components']['shortfall_penalty'] == 3600.0

        # Total revenue should reflect the severity
        # DAM: 25 × 150 = 3750€
        # Balancing: -10 × 155 = -1550€
        # Degradation: 15 × 10 = 150€
        # Shortfall: 3600€
        # Net = 3750 - 1550 - 150 - 3600 = -1550€
        assert result['components']['net_profit'] < 0
