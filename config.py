"""
Configuration for Training/Production mode switching.
"""
from enum import Enum


class Mode(Enum):
    TRAINING = "training"
    PRODUCTION = "production"


# ============================================================================
# CURRENT MODE (Change this to switch between Training and Production)
# ============================================================================
CURRENT_MODE = Mode.TRAINING


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_data_provider(df=None, battery_params=None, **kwargs):
    """
    Factory function to create the appropriate data provider.
    
    Args:
        df: DataFrame for training mode (required if TRAINING)
        battery_params: Battery configuration dict
        **kwargs: Additional arguments for specific providers
        
    Returns:
        IDataProvider instance
    """
    from data.data_provider import SimulatedDataProvider, LiveDataProvider
    
    if CURRENT_MODE == Mode.TRAINING:
        if df is None:
            raise ValueError("DataFrame required for TRAINING mode")
        if battery_params is None:
            battery_params = {
                'capacity_mwh': 146.0,      # Updated: 146 MWh capacity (was 30.0)
                'max_discharge_mw': 30.0,   # Inverter rating: 30 MW (unchanged)
                'efficiency': 0.94
            }
        return SimulatedDataProvider(df, battery_params, **kwargs)
    
    elif CURRENT_MODE == Mode.PRODUCTION:
        # In production, you would pass API clients here
        market_api = kwargs.get('market_api_client')
        battery_plc = kwargs.get('battery_plc_client')
        forecast_svc = kwargs.get('forecast_service')
        
        return LiveDataProvider(
            market_api_client=market_api,
            battery_plc_client=battery_plc,
            forecast_service=forecast_svc
        )
    
    else:
        raise ValueError(f"Unknown mode: {CURRENT_MODE}")
