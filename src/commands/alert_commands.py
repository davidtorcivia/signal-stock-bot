"""
Alert command for price notifications.

Provides: !alert
"""

from .base import BaseCommand, CommandContext, CommandResult
from ..database import AlertsDB, hash_phone
from ..providers import ProviderManager


class AlertCommand(BaseCommand):
    """Manage price alerts."""
    name = "alert"
    aliases = ["alerts", "notify"]
    description = "Price alert notifications"
    usage = "!alert AAPL above 200"
    help_explanation = """Set alerts to be notified when prices hit targets.

**Commands:**
- !alert AAPL above 200 — Notify when AAPL > $200
- !alert TSLA below 180 — Notify when TSLA < $180
- !alert BTC change 5 — Notify on 5% move
- !alerts — List your active alerts
- !alert remove 1 — Remove alert by ID
- !alert clear — Clear all your alerts

**Notes:**
- Alerts notify in the same context where set (DM or group)
- Limit: 20 active alerts per user
- Checked every 60 seconds"""
    
    def __init__(self, provider_manager: ProviderManager, alerts_db: AlertsDB):
        self.providers = provider_manager
        self.db = alerts_db
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        user_hash = hash_phone(ctx.sender)
        
        if not ctx.args:
            return await self._list_alerts(user_hash)
        
        # Parse subcommand
        first_arg = ctx.args[0].lower()
        
        if first_arg in ("list", "show"):
            return await self._list_alerts(user_hash)
        elif first_arg in ("remove", "delete", "rm"):
            return await self._remove_alert(user_hash, ctx.args[1:])
        elif first_arg == "clear":
            return await self._clear_alerts(user_hash)
        else:
            # Try to parse as: SYMBOL CONDITION VALUE
            return await self._add_alert(ctx, user_hash)
    
    async def _list_alerts(self, user_hash: str) -> CommandResult:
        """List all active alerts."""
        alerts = await self.db.get_active_alerts(user_hash)
        
        if not alerts:
            return CommandResult.ok(
                "◈ Your Alerts\n\n"
                "No active alerts.\n"
                "Add with: !alert AAPL above 200"
            )
        
        lines = ["◈ Your Alerts", ""]
        
        for alert in alerts:
            symbol = alert["symbol"]
            condition = alert["condition"]
            target = alert["target_value"]
            alert_id = alert["id"]
            
            if condition == "above":
                indicator = "▲"
                desc = f"above ${target:.2f}"
            elif condition == "below":
                indicator = "▼"
                desc = f"below ${target:.2f}"
            else:  # change_pct
                indicator = "◇"
                desc = f"moves {target:.1f}%"
            
            context = "group" if alert.get("group_id") else "DM"
            lines.append(f"{indicator} [{alert_id}] {symbol} {desc} ({context})")
        
        lines.append(f"\n({len(alerts)}/{self.db.MAX_ALERTS_PER_USER} alerts)")
        
        return CommandResult.ok("\n".join(lines))
    
    async def _add_alert(self, ctx: CommandContext, user_hash: str) -> CommandResult:
        """Add a new price alert."""
        args = ctx.args
        
        if len(args) < 3:
            return CommandResult.error(
                "Format: !alert SYMBOL CONDITION VALUE\n"
                "Example: !alert AAPL above 200"
            )
        
        # Parse arguments
        from ..utils import resolve_symbol
        symbol, _ = await resolve_symbol(args[0])
        condition = args[1].lower()
        
        try:
            value = float(args[2].replace("$", "").replace("%", ""))
        except ValueError:
            return CommandResult.error(f"Invalid value: {args[2]}")
        
        # Validate condition
        if condition not in ("above", "below", "change"):
            return CommandResult.error(
                "Condition must be: above, below, or change\n"
                "Example: !alert AAPL above 200"
            )
        
        if condition == "change":
            condition = "change_pct"
        
        # Get current price for reference
        try:
            quote = await self.providers.get_quote(symbol)
            current_price = quote.price
        except Exception:
            current_price = None
        
        # Add alert
        alert_id = await self.db.add_alert(
            user_hash=user_hash,
            user_phone=ctx.sender,
            symbol=symbol,
            condition=condition,
            target_value=value,
            group_id=ctx.group_id,
        )
        
        if alert_id is None:
            return CommandResult.error(
                f"Alert limit reached ({self.db.MAX_ALERTS_PER_USER}). "
                "Remove some with !alert remove ID"
            )
        
        # Build confirmation
        if condition == "above":
            desc = f"rises above ${value:.2f}"
        elif condition == "below":
            desc = f"drops below ${value:.2f}"
        else:
            desc = f"moves {value:.1f}%"
        
        context = "this group" if ctx.group_id else "DM"
        lines = [
            f"Alert set for {symbol}",
            f"Trigger when: {desc}",
            f"Notify: {context}",
        ]
        
        if current_price:
            lines.append(f"Current: ${current_price:.2f}")
        
        lines.append(f"\nAlert ID: {alert_id}")
        
        return CommandResult.ok("\n".join(lines))
    
    async def _remove_alert(self, user_hash: str, args: list[str]) -> CommandResult:
        """Remove an alert by ID."""
        if not args:
            return CommandResult.error("Specify alert ID: !alert remove 1")
        
        try:
            alert_id = int(args[0])
        except ValueError:
            return CommandResult.error(f"Invalid alert ID: {args[0]}")
        
        removed = await self.db.remove_alert(alert_id, user_hash)
        
        if removed:
            return CommandResult.ok(f"Alert {alert_id} removed")
        else:
            return CommandResult.error(f"Alert {alert_id} not found (or not yours)")
    
    async def _clear_alerts(self, user_hash: str) -> CommandResult:
        """Clear all alerts."""
        count = await self.db.clear_user_alerts(user_hash)
        
        if count:
            return CommandResult.ok(f"Cleared {count} alert(s)")
        else:
            return CommandResult.ok("No alerts to clear")
