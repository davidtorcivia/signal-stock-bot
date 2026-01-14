"""
Admin commands for bot management.

Provides: !metrics, !cache, !admin
"""

import json
from datetime import datetime
from .base import BaseCommand, CommandContext, CommandResult
from ..cache import get_cache_manager, get_metrics


class MetricsCommand(BaseCommand):
    """Admin command for viewing bot metrics."""
    name = "metrics"
    aliases = ["stats", "perf"]
    description = "Bot performance metrics"
    usage = "!metrics"
    help_explanation = "Shows system health stats like Uptime, Requests Per Minute, Cache Hit Rate, and API Provider Status."
    
    def __init__(self, admin_numbers: list[str] = None):
        """
        Initialize metrics command.
        
        Args:
            admin_numbers: Phone numbers allowed to use this command.
                         If empty/None, anyone can use it.
        """
        self.admin_numbers = admin_numbers or []
    
    def _is_admin(self, sender: str) -> bool:
        """Check if sender is an admin."""
        if not self.admin_numbers:
            return True  # No restrictions
        return sender in self.admin_numbers
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        # Check admin access (if configured)
        # Check admin access (if configured)
        if not self._is_admin(ctx.sender):
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Admin access denied for {ctx.sender}. Allowed: {self.admin_numbers}")
            return CommandResult.error("This command requires admin access.")
        
        metrics = get_metrics()
        stats = metrics.get_all_stats()
        
        # Format uptime
        uptime = stats["uptime_seconds"]
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        uptime_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
        
        lines = [
            "◈ Bot Metrics Dashboard",
            "",
            f"Uptime: {uptime_str}",
            f"Requests/min: {stats['requests_per_minute']:.0f}",
            "",
            "━━━ Cache ━━━",
            f"Hit Rate: {stats['cache']['overall_hit_rate']}",
        ]
        
        # Cache details
        if stats['cache']['caches']:
            for name, cache_stats in stats['cache']['caches'].items():
                hit_rate = cache_stats.get('hit_rate', 0)
                size = cache_stats.get('size', 0)
                lines.append(f"  {name}: {size} entries ({hit_rate:.0f}% hit)")
        
        lines.append("")
        lines.append("━━━ Providers ━━━")
        
        # Provider details
        if stats['providers']:
            for name, prov_stats in stats['providers'].items():
                health = "●" if prov_stats.get('healthy', True) else "○"
                success = prov_stats.get('success_rate', '100%')
                latency = prov_stats.get('avg_latency_ms', '0ms')
                lines.append(f"  {health} {name}: {success} ({latency})")
        else:
            lines.append("  No provider data yet")
        
        return CommandResult.ok("\n".join(lines))


class CacheCommand(BaseCommand):
    """Admin command for cache management."""
    name = "cache"
    aliases = []
    description = "Cache management"
    usage = "!cache [clear|stats]"
    help_explanation = "Manage the internal data cache. View statistics with !cache stats or wipe all data with !cache clear."
    
    def __init__(self, admin_numbers: list[str] = None):
        self.admin_numbers = admin_numbers or []
    
    def _is_admin(self, sender: str) -> bool:
        if not self.admin_numbers:
            return True
        return sender in self.admin_numbers
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        # Check admin access
        if not self._is_admin(ctx.sender):
            return CommandResult.error("This command requires admin access.")
        
        action = ctx.args[0].lower() if ctx.args else "stats"
        cache_mgr = get_cache_manager()
        
        if action == "clear":
            cache_mgr.clear_all()
            return CommandResult.ok("All caches cleared.")
        
        elif action == "stats":
            stats = cache_mgr.get_all_stats()
            
            lines = ["◈ Cache Statistics", ""]
            
            for name, cache_stats in stats.items():
                size = cache_stats.get("size", 0)
                hits = cache_stats.get("hits", 0)
                misses = cache_stats.get("misses", 0)
                ttl = cache_stats.get("ttl_seconds", 0)
                
                total = hits + misses
                rate = (hits / total * 100) if total > 0 else 0
                
                lines.append(f"{name} (TTL: {ttl}s)")
                lines.append(f"  Entries: {size} | Hits: {hits} | Misses: {misses}")
                lines.append(f"  Hit Rate: {rate:.1f}%")
                lines.append("")
            
            return CommandResult.ok("\n".join(lines))
        
        else:
            return CommandResult.error(f"Unknown action: {action}\nUsage: {self.usage}")


class AdminCommand(BaseCommand):
    """Admin-only commands for bot management."""
    name = "admin"
    aliases = []
    description = "Admin utilities"
    usage = "!admin [backup|alerts|users]"
    help_explanation = """Admin-only utilities:

**Commands:**
• !admin backup — Export all watchlists as JSON (sent as DM)
• !admin alerts — List all active alerts across users
• !admin users — Show user count and activity"""
    
    def __init__(self, admin_numbers: list[str], watchlist_db=None, alerts_db=None):
        self.admin_numbers = admin_numbers
        self.watchlist_db = watchlist_db
        self.alerts_db = alerts_db
    
    def _is_admin(self, sender: str) -> bool:
        if not self.admin_numbers:
            return False  # Admin command requires explicit list
        return sender in self.admin_numbers
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        if not self._is_admin(ctx.sender):
            return CommandResult.error("This command requires admin access.")
        
        if not ctx.args:
            return CommandResult.ok(
                "◈ Admin Commands\n\n"
                "!admin backup — Export watchlists\n"
                "!admin alerts — List all alerts\n"
                "!admin users — User activity"
            )
        
        action = ctx.args[0].lower()
        
        if action == "backup":
            return await self._backup(ctx)
        elif action == "alerts":
            return await self._list_alerts(ctx)
        elif action == "users":
            return await self._user_stats(ctx)
        else:
            return CommandResult.error(f"Unknown action: {action}")
    
    async def _backup(self, ctx: CommandContext) -> CommandResult:
        """Export all watchlists as JSON."""
        if not self.watchlist_db:
            return CommandResult.error("Database not configured")
        
        try:
            # Get all data from DB (admin only feature)
            import aiosqlite
            async with aiosqlite.connect(self.watchlist_db.db_path) as db:
                cursor = await db.execute(
                    "SELECT user_hash, symbol, created_at FROM watchlist ORDER BY user_hash, symbol"
                )
                rows = await cursor.fetchall()
            
            # Group by user
            data = {}
            for user_hash, symbol, created_at in rows:
                short_hash = user_hash[:8]  # Truncate for privacy
                if short_hash not in data:
                    data[short_hash] = []
                data[short_hash].append(symbol)
            
            export = {
                "exported_at": datetime.utcnow().isoformat(),
                "total_users": len(data),
                "total_symbols": len(rows),
                "watchlists": data,
            }
            
            result = CommandResult.ok(
                f"◈ Watchlist Backup\n\n"
                f"Users: {len(data)}\n"
                f"Total symbols: {len(rows)}\n\n"
                f"```json\n{json.dumps(export, indent=2)[:2000]}\n```"
            )
            result.dm_only = True
            return result
            
        except Exception as e:
            return CommandResult.error(f"Backup failed: {e}")
    
    async def _list_alerts(self, ctx: CommandContext) -> CommandResult:
        """List all active alerts."""
        if not self.alerts_db:
            return CommandResult.ok("◈ Alerts\n\nNo alerts configured.")
        
        # TODO: Implement when AlertsDB is created
        return CommandResult.ok("◈ Alerts\n\nAlert system not yet implemented.")
    
    async def _user_stats(self, ctx: CommandContext) -> CommandResult:
        """Show user statistics."""
        if not self.watchlist_db:
            return CommandResult.error("Database not configured")
        
        try:
            import aiosqlite
            async with aiosqlite.connect(self.watchlist_db.db_path) as db:
                # Count unique users
                cursor = await db.execute("SELECT COUNT(DISTINCT user_hash) FROM watchlist")
                user_count = (await cursor.fetchone())[0]
                
                # Count total symbols
                cursor = await db.execute("SELECT COUNT(*) FROM watchlist")
                symbol_count = (await cursor.fetchone())[0]
                
                # Average symbols per user
                avg = symbol_count / user_count if user_count > 0 else 0
            
            return CommandResult.ok(
                f"◈ User Statistics\n\n"
                f"Total users: {user_count}\n"
                f"Total watchlist symbols: {symbol_count}\n"
                f"Avg symbols/user: {avg:.1f}"
            )
            
        except Exception as e:
            return CommandResult.error(f"Stats failed: {e}")
