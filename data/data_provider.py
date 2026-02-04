"""
Data Provider Pattern - Dependency Inversion for Training/Production Flexibility.

This module defines the interface and implementations for data sources.
The Agent/Environment never knows where data comes from - it just asks the provider.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
import datetime
import numpy as np
import pandas as pd


# ============================================================================
# DATA TRANSFER OBJECTS (DTOs)
# ============================================================================

@dataclass
class MarketState:
    """
    Represents the current market state at a given timestamp.
    This is the COMMON language between Training and Production.
    """
    timestamp: datetime.datetime
    dam_price: float              # DAM price (€/MWh) - locked day-ahead
    intraday_price: float         # Current Intraday mid-price (€/MWh)
    dam_commitment_mw: float      # + for sell, - for buy (charge)
    
    # Order Book (Bid-Ask)
    best_bid: float = 0.0         # Price to SELL at immediately
    best_ask: float = 0.0         # Price to BUY at immediately
    bid_volume: float = 50.0      # Max MW at best_bid
    ask_volume: float = 50.0      # Max MW at best_ask
    spread: float = 0.0           # ask - bid
    
    # Imbalance prices (for counter-trading)
    imbalance_price_long: float = 0.0   # Price for buying (covering shortage)
    imbalance_price_short: float = 0.0  # Price for selling (excess)
    
    # Time features (for observation)
    hour_sin: float = 0.0
    hour_cos: float = 0.0


@dataclass
class BatteryTelemetry:
    """Battery state: simulated (training) or real (production)."""
    soc: float                    # State of Charge (0.0 - 1.0)
    max_charge_mw: float          # Available charge capacity (MW)
    max_discharge_mw: float       # Available discharge capacity (MW)
    health_status: bool = True    # True = OK, False = Fault


@dataclass
class PriceForecast:
    """Price forecast for lookahead."""
    prices: List[float]           # List of prices for next N hours
    dam_commitments: List[float]  # List of DAM commitments for next N hours


@dataclass
class NormalizationConfig:
    """Normalization parameters for observation scaling."""
    price_scale: float = 100.0     # €/MWh scale
    power_scale: float = 50.0      # MW scale
    soc_scale: float = 1.0         # SoC is already 0-1


# ============================================================================
# INTERFACE (Abstract Base Class)
# ============================================================================

class IDataProvider(ABC):
    """The contract. Environment talks ONLY to this interface."""
    
    @abstractmethod
    def get_current_market_state(self) -> MarketState:
        pass

    @abstractmethod
    def get_battery_telemetry(self) -> BatteryTelemetry:
        pass

    @abstractmethod
    def get_forecast(self, hours_ahead: int) -> PriceForecast:
        pass
    
    @abstractmethod
    def dispatch_battery(self, mw_setpoint: float) -> None:
        pass
    
    @abstractmethod
    def step(self) -> bool:
        pass
    
    @abstractmethod
    def reset(self) -> None:
        pass
    
    @abstractmethod
    def get_max_steps(self) -> int:
        pass
    
    @abstractmethod
    def get_normalization_config(self) -> NormalizationConfig:
        pass


# ============================================================================
# TRAINING IMPLEMENTATION (Simulated Data from CSV/DataFrame)
# ============================================================================

class SimulatedDataProvider(IDataProvider):
    """Data provider for Training mode with Market Depth simulation."""
    
    def __init__(
        self, 
        df: pd.DataFrame,
        battery_params: Dict,
        initial_soc: float = 0.5,
        lookahead_hours: int = 4,
        noise_std: float = 5.0,
        base_spread: float = 2.0,
        base_volume: float = 50.0
    ):
        self.df = df.copy()
        self.df.columns = [c.lower() for c in self.df.columns]
        self.battery_params = battery_params
        self.initial_soc = initial_soc
        self.lookahead_hours = lookahead_hours
        self.noise_std = noise_std
        self.base_spread = base_spread
        self.base_volume = base_volume
        
        # Pre-compute volatility for spread simulation
        prices = self.df['price'].values if 'price' in self.df.columns else np.ones(len(self.df)) * 100
        self.price_volatility = pd.Series(prices).rolling(24, min_periods=1).std().fillna(5.0).values
        
        # Battery parameters
        self.capacity_mwh = battery_params.get('capacity_mwh', 146.0)
        self.max_power_mw = battery_params.get('max_discharge_mw', 30.0)
        self.efficiency = battery_params.get('efficiency', 0.94)
        self.min_soc = 0.05
        self.max_soc = 0.95
        
        # Normalization config
        self.norm_config = NormalizationConfig(
            price_scale=100.0,
            power_scale=self.max_power_mw
        )
        
        # State
        self.current_step = 0
        self.current_soc = initial_soc
        self.max_steps = len(df) - 1 - lookahead_hours
        
    def get_current_market_state(self) -> MarketState:
        row = self.df.iloc[self.current_step]
        
        # DAM price
        dam_price = row.get('price', 100.0)
        
        # Intraday price with noise
        if 'intraday_price' in row and pd.notna(row.get('intraday_price')):
            intraday_price = row['intraday_price']
        else:
            noise = np.random.normal(0, self.noise_std) if self.noise_std > 0 else 0
            intraday_price = max(0.1, dam_price + noise)
        
        # --- BID-ASK SPREAD SIMULATION ---
        volatility = self.price_volatility[self.current_step]
        spread = self.base_spread * (1 + volatility / 20)
        spread = min(spread, 20.0)
        
        best_bid = intraday_price - spread / 2
        best_ask = intraday_price + spread / 2
        
        # Volume simulation
        bid_volume = self.base_volume * np.random.uniform(0.5, 1.5)
        ask_volume = self.base_volume * np.random.uniform(0.5, 1.5)
        
        # Imbalance prices
        long_price = row.get('long', best_ask * 1.05)
        short_price = row.get('short', best_bid * 0.95)
        
        # Other fields
        dam_commitment = row.get('dam_commitment', 0.0)
        hour_sin = row.get('hour_sin', 0.0)
        hour_cos = row.get('hour_cos', 0.0)
        
        timestamp = row.name if hasattr(row, 'name') and isinstance(row.name, datetime.datetime) else datetime.datetime.now()
        
        return MarketState(
            timestamp=timestamp,
            dam_price=dam_price,
            intraday_price=intraday_price,
            dam_commitment_mw=dam_commitment,
            best_bid=best_bid,
            best_ask=best_ask,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            spread=spread,
            imbalance_price_long=long_price,
            imbalance_price_short=short_price,
            hour_sin=hour_sin,
            hour_cos=hour_cos
        )
    
    def get_battery_telemetry(self) -> BatteryTelemetry:
        eff_sqrt = np.sqrt(self.efficiency)
        
        available_discharge_mwh = max(0.0, (self.current_soc - self.min_soc) * self.capacity_mwh)
        available_charge_mwh = max(0.0, (self.max_soc - self.current_soc) * self.capacity_mwh)
        
        max_discharge_mw = min(self.max_power_mw, available_discharge_mwh * eff_sqrt)
        max_charge_mw = min(self.max_power_mw, available_charge_mwh / eff_sqrt)
        
        return BatteryTelemetry(
            soc=self.current_soc,
            max_charge_mw=max_charge_mw,
            max_discharge_mw=max_discharge_mw,
            health_status=True
        )
    
    def get_forecast(self, hours_ahead: int) -> PriceForecast:
        prices = []
        dam_commitments = []
        
        current_row = self.df.iloc[self.current_step]
        current_price = current_row.get('price', 100.0)
        current_dam = current_row.get('dam_commitment', 0.0)
        
        for i in range(1, hours_ahead + 1):
            if self.current_step + i < len(self.df):
                future_row = self.df.iloc[self.current_step + i]
                prices.append(future_row.get('price', current_price))
                dam_commitments.append(future_row.get('dam_commitment', current_dam))
            else:
                prices.append(current_price)
                dam_commitments.append(current_dam)
        
        return PriceForecast(prices=prices, dam_commitments=dam_commitments)
    
    def dispatch_battery(self, mw_setpoint: float) -> None:
        eff_sqrt = np.sqrt(self.efficiency)
        
        if mw_setpoint < 0:  # Charging
            energy_in = abs(mw_setpoint) * eff_sqrt
            self.current_soc += energy_in / self.capacity_mwh
        else:  # Discharging
            energy_out = mw_setpoint / eff_sqrt
            self.current_soc -= energy_out / self.capacity_mwh
        
        self.current_soc = np.clip(self.current_soc, 0.0, 1.0)
    
    def step(self) -> bool:
        self.current_step += 1
        return self.current_step < self.max_steps
    
    def reset(self) -> None:
        self.current_step = 0
        self.current_soc = self.initial_soc
    
    def get_max_steps(self) -> int:
        return self.max_steps
    
    def get_normalization_config(self) -> NormalizationConfig:
        return self.norm_config


# ============================================================================
# PRODUCTION IMPLEMENTATION (Live Data from ENTSO-E APIs)
# ============================================================================

class LiveDataProvider(IDataProvider):
    """
    Production mode: Connects to ENTSO-E API for real-time market data.
    
    Features:
    - Live DAM prices, wind/solar forecasts, imbalance prices from ENTSO-E
    - Rolling buffer for calculating imbalance risk features
    - Battery telemetry placeholder (connect to real PLC)
    
    Usage:
        provider = LiveDataProvider(
            api_key="your-entsoe-key",
            battery_params={'capacity_mwh': 50, 'max_discharge_mw': 30}
        )
        market_state = provider.get_current_market_state()
        imbalance_info = provider.get_imbalance_risk()
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        battery_params: Optional[Dict] = None,
        initial_soc: float = 0.5,
        buffer_hours: int = 24,  # Rolling buffer size for calculations
        time_step_hours: float = 1.0
    ):
        import os
        from collections import deque
        
        self.api_key = api_key or os.getenv('ENTSOE_API_KEY')
        if not self.api_key:
            raise ValueError("ENTSOE_API_KEY must be provided or set in environment variables.")
        
        # Initialize ENTSO-E connector
        from data.entsoe_connector import EntsoeConnector
        self.entsoe = EntsoeConnector(api_key=self.api_key)
        
        # Battery parameters
        self.battery_params = battery_params or {'capacity_mwh': 146.0, 'max_discharge_mw': 30.0}
        self.capacity_mwh = self.battery_params.get('capacity_mwh', 146.0)
        self.max_power_mw = self.battery_params.get('max_discharge_mw', 30.0)
        self.efficiency = self.battery_params.get('efficiency', 0.94)
        self.min_soc = 0.05
        self.max_soc = 0.95
        
        # Time configuration
        self.time_step_hours = time_step_hours
        self.steps_per_hour = int(1.0 / time_step_hours)
        
        # Action space configuration (for masking)
        self.n_actions = 21
        import numpy as np
        self.action_levels = np.linspace(-1.0, 1.0, self.n_actions)
        
        # Rolling buffers for imbalance risk calculation
        buffer_size = int(buffer_hours / time_step_hours)
        self.price_buffer = deque(maxlen=buffer_size)
        self.res_buffer = deque(maxlen=buffer_size)  # solar + wind
        
        # State
        self.current_soc = initial_soc
        self._last_fetch_time = None
        self._cached_market_data = None
        self._cache_ttl_minutes = 5  # Refresh every 5 minutes
        
        # Normalization config
        self.norm_config = NormalizationConfig(
            price_scale=100.0,
            power_scale=self.max_power_mw
        )
        
        print(f"LiveDataProvider initialized: buffer={buffer_hours}h, time_step={time_step_hours}h")
    
    def _fetch_latest_data(self) -> Dict:
        """Fetch latest market data from ENTSO-E."""
        now = pd.Timestamp.now(tz='Europe/Athens')
        
        # Check cache validity
        if self._cached_market_data and self._last_fetch_time:
            elapsed = (now - self._last_fetch_time).total_seconds() / 60
            if elapsed < self._cache_ttl_minutes:
                return self._cached_market_data
        
        # Fetch fresh data
        start = now.floor('D') - pd.Timedelta(days=1)
        end = now.floor('D') + pd.Timedelta(days=2)
        
        try:
            # DAM prices
            dam_df = self.entsoe.fetch_day_ahead_prices(start, end)
            
            # Wind/Solar forecasts
            res_df = self.entsoe.fetch_wind_solar_forecast(start, end)
            
            # Imbalance prices
            imb_df = self.entsoe.fetch_imbalance_prices(start, end)
            
            # Merge all data
            data = {
                'dam_prices': dam_df,
                'res_forecast': res_df,
                'imbalance': imb_df,
                'fetch_time': now
            }
            
            # Calculate Daily Stats for DAM Commitment Heuristic
            if not dam_df.empty:
                # Filter for *today* to calculate today's median
                today = now.floor('D')
                today_mask = (dam_df.index >= today) & (dam_df.index < today + pd.Timedelta(days=1))
                today_prices = dam_df.loc[today_mask, 'price']
                
                if not today_prices.empty:
                    data['daily_median'] = today_prices.median()
                    data['daily_max'] = today_prices.max()
                    data['daily_min'] = today_prices.min()
            
            self._cached_market_data = data
            self._last_fetch_time = now
            
            # Update rolling buffers
            self._update_buffers(dam_df, res_df)
            
            return data
            
        except Exception as e:
            print(f"Error fetching ENTSO-E data: {e}")
            return self._cached_market_data or {}
    
    def _update_buffers(self, dam_df: pd.DataFrame, res_df: pd.DataFrame):
        """Update rolling buffers with latest data."""
        if dam_df is not None and not dam_df.empty:
            for price in dam_df['price'].tail(self.steps_per_hour * 2).values:
                self.price_buffer.append(price)
        
        if res_df is not None and not res_df.empty:
            # Sum solar + wind
            if 'solar' in res_df.columns and 'wind_onshore' in res_df.columns:
                total_res = (res_df['solar'] + res_df['wind_onshore']).tail(self.steps_per_hour * 2)
                for val in total_res.values:
                    self.res_buffer.append(val)
    
    def get_imbalance_risk(self) -> Dict:
        """
        Calculate real-time imbalance risk using rolling buffers.
        
        Returns:
            dict with 'gradient', 'res_deviation', 'risk_score'
        """
        import numpy as np
        
        # Default values if not enough data
        if len(self.price_buffer) < 2:
            return {'gradient': 0.0, 'res_deviation': 0.0, 'risk_score': 0.0}
        
        # 1. Net Load Gradient (price change per hour)
        prices = list(self.price_buffer)
        if len(prices) >= self.steps_per_hour:
            gradient = prices[-1] - prices[-self.steps_per_hour]
        else:
            gradient = prices[-1] - prices[0]
        
        # 2. RES Deviation Trend
        res_deviation = 0.0
        if len(self.res_buffer) >= 2:
            res_values = list(self.res_buffer)
            current_res = res_values[-1]
            avg_res = np.mean(res_values[:-1]) if len(res_values) > 1 else current_res
            res_deviation = current_res - avg_res
        
        # 3. Risk Score Calculation
        risk_score = 0.0
        
        # Factor 1: Price gradient
        if gradient > 20:
            risk_score += min(0.4, gradient / 50.0)
        
        # Factor 2: Time of day (sunset risk)
        now = datetime.datetime.now()
        if 17 <= now.hour <= 20:
            risk_score += 0.3
        
        # Factor 3: Current price level
        if len(prices) > 0 and prices[-1] > 150:
            risk_score += min(0.3, (prices[-1] - 150) / 100.0)
        
        # Factor 4: RES underperformance
        if res_deviation < -50:
            risk_score += min(0.2, abs(res_deviation) / 200.0)
        
        return {
            'gradient': gradient,
            'res_deviation': res_deviation,
            'risk_score': min(risk_score, 1.0)
        }
    
    def get_current_market_state(self) -> MarketState:
        """Get current market state from ENTSO-E data."""
        data = self._fetch_latest_data()
        now = pd.Timestamp.now(tz='Europe/Athens')
        
        # Get current hour data
        dam_df = data.get('dam_prices', pd.DataFrame())
        imb_df = data.get('imbalance', pd.DataFrame())
        
        # DAM price
        dam_price = 100.0  # Default
        if not dam_df.empty:
            # Find closest timestamp
            closest_idx = dam_df.index.get_indexer([now], method='nearest')[0]
            if closest_idx >= 0 and closest_idx < len(dam_df):
                dam_price = dam_df.iloc[closest_idx]['price']
        
        # Intraday price (approximate as DAM + noise for now)
        intraday_price = dam_price + np.random.normal(0, 5)
        
        # Bid-Ask spread simulation
        spread = 4.0  # Base spread
        best_bid = max(0.1, intraday_price - spread / 2)
        best_ask = intraday_price + spread / 2
        
        # Imbalance prices
        long_price = 0.0
        short_price = 0.0
        if not imb_df.empty:
            closest_idx = imb_df.index.get_indexer([now], method='nearest')[0]
            if closest_idx >= 0 and closest_idx < len(imb_df):
                row = imb_df.iloc[closest_idx]
                long_price = row.get('Long', best_ask * 1.05)
                short_price = row.get('Short', best_bid * 0.95)
        
        # SIMULATED DAM SCHEDULER (Heuristic)
        # Strategy: Discharge when price > daily_median * 1.1, Charge when < 0.9
        dam_commitment_mw = 0.0
        daily_median = data.get('daily_median')
        
        if daily_median is not None:
            eff_sqrt = np.sqrt(self.efficiency)
            # Assuming 1h max power for simplicity of commitment
            max_c = self.max_power_mw
            
            if dam_price > daily_median * 1.1:
                # High price: Commit to Discharge (e.g. 50% capacity)
                dam_commitment_mw = max_c * 0.5
            elif dam_price < daily_median * 0.9:
                # Low price: Commit to Charge
                dam_commitment_mw = -max_c * 0.5
                
        # Time encoding
        hour = now.hour + now.minute / 60.0
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        return MarketState(
            timestamp=now.to_pydatetime(),
            dam_price=dam_price,
            intraday_price=intraday_price,
            dam_commitment_mw=dam_commitment_mw, # Updated form Heuristic
            best_bid=best_bid,
            best_ask=best_ask,
            bid_volume=50.0,
            ask_volume=50.0,
            spread=spread,
            imbalance_price_long=long_price,
            imbalance_price_short=short_price,
            hour_sin=hour_sin,
            hour_cos=hour_cos
        )
    
    def get_action_mask(self) -> np.ndarray:
        """Returns boolean mask of valid actions based on current SoC."""
        eff_sqrt = np.sqrt(self.efficiency)
        
        # Available energy in MWh
        available_discharge_mwh = max(0.0, (self.current_soc - self.min_soc) * self.capacity_mwh)
        available_charge_mwh = max(0.0, (self.max_soc - self.current_soc) * self.capacity_mwh)
        
        # Max power considering what can be delivered in this time step
        # Energy per step = Power × time_step_hours, so max_power = energy / time_step_hours
        max_discharge_mw = min(self.max_power_mw, (available_discharge_mwh * eff_sqrt) / self.time_step_hours)
        max_charge_mw = min(self.max_power_mw, (available_charge_mwh / eff_sqrt) / self.time_step_hours)
        
        max_discharge_action = max_discharge_mw / self.max_power_mw
        max_charge_action = max_charge_mw / self.max_power_mw
        
        mask = np.zeros(self.n_actions, dtype=bool)
        for i, action_level in enumerate(self.action_levels):
            if action_level >= 0:
                mask[i] = action_level <= max_discharge_action + 0.01
            else:
                mask[i] = abs(action_level) <= max_charge_action + 0.01
        
        # Always allow idle
        mask[self.n_actions // 2] = True
        return mask

    def get_battery_telemetry(self) -> BatteryTelemetry:
        """
        Get battery telemetry.
        
        TODO: Connect to real battery SCADA/PLC system.
        Currently returns simulated state.
        """
        eff_sqrt = np.sqrt(self.efficiency)
        
        available_discharge_mwh = max(0.0, (self.current_soc - self.min_soc) * self.capacity_mwh)
        available_charge_mwh = max(0.0, (self.max_soc - self.current_soc) * self.capacity_mwh)
        
        max_discharge_mw = min(self.max_power_mw, available_discharge_mwh * eff_sqrt)
        max_charge_mw = min(self.max_power_mw, available_charge_mwh / eff_sqrt)
        
        return BatteryTelemetry(
            soc=self.current_soc,
            max_charge_mw=max_charge_mw,
            max_discharge_mw=max_discharge_mw,
            health_status=True
        )
    
    def get_forecast(self, hours_ahead: int) -> PriceForecast:
        """Get price forecast from ENTSO-E DAM prices."""
        data = self._fetch_latest_data()
        dam_df = data.get('dam_prices', pd.DataFrame())
        
        now = pd.Timestamp.now(tz='Europe/Athens')
        prices = []
        dam_commitments = []
        
        for i in range(1, hours_ahead + 1):
            future_time = now + pd.Timedelta(hours=i)
            
            if not dam_df.empty:
                closest_idx = dam_df.index.get_indexer([future_time], method='nearest')[0]
                if closest_idx >= 0 and closest_idx < len(dam_df):
                    prices.append(dam_df.iloc[closest_idx]['price'])
                else:
                    prices.append(100.0)
            else:
                prices.append(100.0)
            
            dam_commitments.append(0.0)  # Should come from DAM scheduler
        
        return PriceForecast(prices=prices, dam_commitments=dam_commitments)
    
    def dispatch_battery(self, mw_setpoint: float) -> None:
        """
        Dispatch battery command.
        
        TODO: Connect to real battery SCADA/PLC system.
        Currently simulates SoC change.
        """
        eff_sqrt = np.sqrt(self.efficiency)
        
        if mw_setpoint < 0:  # Charging
            energy_in = abs(mw_setpoint) * self.time_step_hours * eff_sqrt
            self.current_soc += energy_in / self.capacity_mwh
        else:  # Discharging
            energy_out = mw_setpoint * self.time_step_hours / eff_sqrt
            self.current_soc -= energy_out / self.capacity_mwh
        
        self.current_soc = np.clip(self.current_soc, 0.0, 1.0)
        
        print(f"[DISPATCH] {mw_setpoint:+.1f} MW -> SoC: {self.current_soc:.1%}")
    
    def step(self) -> bool:
        """In production, always return True (continuous operation)."""
        return True
    
    def reset(self) -> None:
        """Reset provider state."""
        self.price_buffer.clear()
        self.res_buffer.clear()
        self._cached_market_data = None
        self._last_fetch_time = None
    
    def get_max_steps(self) -> int:
        """In production, infinite steps."""
        return float('inf')
    
    def get_normalization_config(self) -> NormalizationConfig:
        return self.norm_config

