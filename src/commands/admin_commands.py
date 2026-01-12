"""
Admin commands for bot management.

Provides: !metrics, !cache
"""

from datetime import datetime
from .base import BaseCommand, CommandContext, CommandResult
from ..cache import get_cache_manager, get_metrics


class MetricsCommand(BaseCommand):
    """Admin command for viewing bot metrics."""
    name = "metrics"
    aliases = ["stats", "perf"]
    description = "Bot performance metrics (admin)"
    usage = "!metrics"
    
    def __init__(self, admin_numbers: list[str] = None):
        """
        Initialize metrics command.
        
        Args:
            admin_numbers: Phone numbers allowed to use this command.
                         If empty/None, anyone can use it.
        """
        self.admin_numbers = admin_numbers or []
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        # Check admin access (if configured)
        if self.admin_numbers:
            sender = ctx.message.get("envelope", {}).get("source", "")
            if sender not in self.admin_numbers:
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
                health = "✓" if prov_stats.get('healthy', True) else "✗"
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
    description = "Cache management (admin)"
    usage = "!cache [clear|stats]"
    
    def __init__(self, admin_numbers: list[str] = None):
        self.admin_numbers = admin_numbers or []
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        # Check admin access
        if self.admin_numbers:
            sender = ctx.message.get("envelope", {}).get("source", "")
            if sender not in self.admin_numbers:
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
