"""Signal integration package."""

from .handler import SignalHandler, SignalConfig
from .poller import SignalPoller
from .pool import SignalHandlerPool

__all__ = ["SignalHandler", "SignalConfig", "SignalPoller", "SignalHandlerPool"]
