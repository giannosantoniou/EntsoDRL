"""
Market Stage Implementations

I implement the 7 pipeline stages that replace the inline step() code.
Each stage wraps an existing executor and manages ctx state flow.

Pipeline order:
  Pre-pipeline: IdaGateStage, AfrrStage.execute_commitment()
  Priority:     DamStage, AfrrStage.execute()
  Cascade:      IdaStage -> MfrrStage -> XbidStage -> FreeBidStage
"""

from typing import TYPE_CHECKING

from gym_envs.step_context import StepContext
from gym_envs.market_executors import MarketResult

if TYPE_CHECKING:
    from gym_envs.battery_env_unified import BatteryEnvUnified


class IdaGateStage:
    """I trigger IDA gate closures before market execution.

    At gate hours (15:00/22:00/10:00), I generate and lock
    IDA schedules that become mandatory commitments.
    """

    name = 'ida_gate'

    def __init__(self, env: 'BatteryEnvUnified'):
        self._env = env

    def execute(self, ctx: StepContext) -> None:
        self._env._ida_executor.check_gate_triggers(ctx.action)


class AfrrStage:
    """I handle aFRR commitment and energy delivery in two phases.

    Phase 1 (execute_commitment): Pre-DAM commitment, selection, capacity revenue.
    Phase 2 (execute): Post-DAM energy delivery when activated.
    """

    name = 'afrr'

    def __init__(self, env: 'BatteryEnvUnified'):
        self._env = env

    def execute_commitment(self, ctx: StepContext) -> None:
        """I process aFRR commitment at block boundaries and check activation."""
        env = self._env
        afrr_action = ctx.action[0]
        price_tier_action = ctx.action[1]

        # I delegate commitment + selection to the executor
        env._afrr_executor.process_commitment(
            afrr_action, price_tier_action, ctx.remaining_capacity
        )

        # I collect capacity revenue if selected
        ctx.afrr_capacity_revenue = env._afrr_executor.collect_capacity_revenue(ctx.row)

        # I check activation (direction determined here, delivery in execute())
        ctx.afrr_activated, ctx.afrr_direction = env._afrr_executor.check_activation(ctx.row)

    def execute(self, ctx: StepContext) -> None:
        """I deliver aFRR energy after DAM has updated SoC."""
        env = self._env

        if ctx.afrr_activated:
            result = env._afrr_executor.execute_activation(
                ctx.afrr_direction, ctx.remaining_capacity, ctx.row
            )
            ctx.results['afrr'] = result
            ctx.afrr_energy_delivered = result.energy_mw
            ctx.afrr_max_deliverable_mw = result.metadata.get('max_deliverable_mw')
            ctx.actual_energy_mw += result.energy_mw
            # I deduct actual energy delivered from cascade capacity
            ctx.remaining_capacity -= abs(result.energy_mw)
        else:
            env.steps_since_afrr_activation += 1
            ctx.results['afrr'] = MarketResult()
            # I reserve committed capacity even when not activated
            ctx.remaining_capacity = max(0, ctx.remaining_capacity - env.afrr_commitment_mw)


class DamStage:
    """I execute the mandatory DAM commitment.

    DAM is priority 1 — its capacity was already deducted when computing
    remaining_capacity. I apply the pure DamExecutor result to env state.
    """

    name = 'dam'

    def __init__(self, env: 'BatteryEnvUnified'):
        self._env = env

    def execute(self, ctx: StepContext) -> None:
        result = self._env._dam_executor.execute(ctx.dam_commitment)
        self._env._apply_market_result('dam', result)
        ctx.results['dam'] = result
        ctx.dam_executed_mw = result.energy_mw
        ctx.actual_energy_mw += result.energy_mw


class IdaStage:
    """I execute locked IDA schedule positions.

    IDA is mandatory once locked (like DAM). I use remaining capacity
    after DAM and aFRR in the cascade.
    """

    name = 'ida'

    def __init__(self, env: 'BatteryEnvUnified'):
        self._env = env

    def execute(self, ctx: StepContext) -> None:
        result = self._env._ida_executor.execute(ctx.remaining_capacity)
        ctx.results['ida'] = result
        ctx.ida_energy_mw = result.energy_mw
        ctx.ida_shortfall_mw = result.metadata.get('ida_shortfall_mw', 0.0)
        ctx.actual_energy_mw += result.energy_mw
        ctx.remaining_capacity -= abs(result.energy_mw)


class MfrrStage:
    """I execute mFRR trading (TSO-activated, pay-as-cleared, mandatory).

    mFRR sits after IDA in the cascade. It's TSO-mandated, so daily
    cycle counting is exempt.
    """

    name = 'mfrr'

    def __init__(self, env: 'BatteryEnvUnified'):
        self._env = env

    def execute(self, ctx: StepContext) -> None:
        mfrr_qty_action = ctx.action[3]
        result = self._env._trading_executor.execute_mfrr(
            mfrr_qty_action, ctx.remaining_capacity, ctx.row
        )
        ctx.results['mfrr'] = result
        ctx.mfrr_energy_mw = result.energy_mw
        ctx.mfrr_revenue = result.revenue
        ctx.actual_energy_mw += result.energy_mw
        ctx.remaining_capacity -= abs(result.energy_mw)


class XbidStage:
    """I execute XBID continuous intraday trading.

    XBID uses remaining capacity after mFRR. Revenue is split between
    XBID (ISP1 continuous) and IDA auction profits.
    """

    name = 'xbid'

    def __init__(self, env: 'BatteryEnvUnified'):
        self._env = env

    def execute(self, ctx: StepContext) -> None:
        xbid_action = ctx.action[2]
        result = self._env._trading_executor.execute_intraday(
            xbid_action, ctx.remaining_capacity, ctx.row
        )
        ctx.results['xbid'] = result
        ctx.intraday_energy_mw = result.energy_mw
        ctx.intraday_revenue = result.revenue
        ctx.xbid_energy_mw = result.metadata.get('xbid_energy_mw', 0.0)
        ctx.actual_energy_mw += result.energy_mw
        ctx.remaining_capacity -= abs(result.energy_mw)


class FreeBidStage:
    """I execute Free Bid trading (agent-submitted, pay-as-bid, merit-order).

    Free Bids use remaining capacity after all other markets.
    Only active in full-market mode.
    """

    name = 'free_bid'

    def __init__(self, env: 'BatteryEnvUnified'):
        self._env = env

    def execute(self, ctx: StepContext) -> None:
        env = self._env
        freebid_qty_action = ctx.action[4] if env.enable_full_market else 5
        freebid_price_action = ctx.action[5] if env.enable_full_market else 2
        result = env._trading_executor.execute_free_bid(
            freebid_qty_action, freebid_price_action,
            ctx.remaining_capacity, ctx.row
        )
        ctx.results['free_bid'] = result
        ctx.free_bid_energy_mw = result.energy_mw
        ctx.free_bid_activated = result.is_activated
        ctx.agent_bid_price = result.metadata.get('agent_bid_price', 0.0)
        ctx.actual_energy_mw += result.energy_mw
        ctx.remaining_capacity -= abs(result.energy_mw)
