"""
Market Stage Protocol

I define the contract that all market stages must satisfy.
Structural typing — any class with name + execute(ctx) qualifies.
Adding a new market (e.g., FCR) = creating a class + registering it.
"""

from typing import Protocol, runtime_checkable

from gym_envs.step_context import StepContext


@runtime_checkable
class MarketStage(Protocol):
    """I define the interface for pipeline stages.

    Adding a new market (e.g., FCR) = implementing this protocol.
    No step() changes needed — OCP compliance.
    """

    @property
    def name(self) -> str:
        """I return the stage's name for result tracking."""
        ...

    def execute(self, ctx: StepContext) -> None:
        """I execute my market logic, mutating ctx in place."""
        ...
