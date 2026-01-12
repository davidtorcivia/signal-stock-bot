"""
Tests for watchlist functionality.
"""

import pytest
import tempfile
import os

from src.database import WatchlistDB, hash_phone


class TestHashPhone:
    """Tests for phone number hashing"""
    
    def test_hash_is_deterministic(self):
        """Same phone always produces same hash"""
        phone = "+15551234567"
        assert hash_phone(phone) == hash_phone(phone)
    
    def test_different_phones_different_hashes(self):
        """Different phones produce different hashes"""
        hash1 = hash_phone("+15551234567")
        hash2 = hash_phone("+15559876543")
        assert hash1 != hash2
    
    def test_hash_is_sha256(self):
        """Hash is 64 characters (SHA-256 hex)"""
        result = hash_phone("+15551234567")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)


class TestWatchlistDB:
    """Tests for watchlist database operations"""
    
    @pytest.fixture
    def db(self):
        """Create a fresh temp database for each test"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_watchlist.db")
            yield WatchlistDB(db_path)
    
    @pytest.fixture
    def user_hash(self):
        return hash_phone("+15551234567")
    
    @pytest.mark.asyncio
    async def test_add_single_symbol(self, db, user_hash):
        """Test adding a single symbol"""
        added, skipped = await db.add_symbols(user_hash, ["AAPL"])
        assert added == 1
        assert skipped == []
    
    @pytest.mark.asyncio
    async def test_add_multiple_symbols(self, db, user_hash):
        """Test adding multiple symbols"""
        added, skipped = await db.add_symbols(user_hash, ["AAPL", "MSFT", "GOOGL"])
        assert added == 3
        assert skipped == []
    
    @pytest.mark.asyncio
    async def test_add_duplicate_symbol(self, db, user_hash):
        """Test that duplicates are ignored"""
        await db.add_symbols(user_hash, ["AAPL"])
        added, skipped = await db.add_symbols(user_hash, ["AAPL"])
        assert added == 0
        assert skipped == []
    
    @pytest.mark.asyncio
    async def test_get_watchlist(self, db, user_hash):
        """Test retrieving watchlist"""
        await db.add_symbols(user_hash, ["AAPL", "MSFT"])
        symbols = await db.get_watchlist(user_hash)
        assert set(symbols) == {"AAPL", "MSFT"}
    
    @pytest.mark.asyncio
    async def test_get_empty_watchlist(self, db, user_hash):
        """Test empty watchlist returns empty list"""
        symbols = await db.get_watchlist(user_hash)
        assert symbols == []
    
    @pytest.mark.asyncio
    async def test_remove_symbol(self, db, user_hash):
        """Test removing a symbol"""
        await db.add_symbols(user_hash, ["AAPL", "MSFT"])
        removed = await db.remove_symbol(user_hash, "AAPL")
        assert removed is True
        symbols = await db.get_watchlist(user_hash)
        assert symbols == ["MSFT"]
    
    @pytest.mark.asyncio
    async def test_remove_nonexistent_symbol(self, db, user_hash):
        """Test removing a symbol that doesn't exist"""
        removed = await db.remove_symbol(user_hash, "AAPL")
        assert removed is False
    
    @pytest.mark.asyncio
    async def test_clear_watchlist(self, db, user_hash):
        """Test clearing entire watchlist"""
        await db.add_symbols(user_hash, ["AAPL", "MSFT", "GOOGL"])
        count = await db.clear(user_hash)
        assert count == 3
        symbols = await db.get_watchlist(user_hash)
        assert symbols == []
    
    @pytest.mark.asyncio
    async def test_clear_empty_watchlist(self, db, user_hash):
        """Test clearing already empty watchlist"""
        count = await db.clear(user_hash)
        assert count == 0
    
    @pytest.mark.asyncio
    async def test_count(self, db, user_hash):
        """Test counting symbols"""
        await db.add_symbols(user_hash, ["AAPL", "MSFT"])
        count = await db.count(user_hash)
        assert count == 2
    
    @pytest.mark.asyncio
    async def test_symbol_limit(self, db, user_hash):
        """Test that symbol limit is enforced"""
        # Add 50 symbols (the limit)
        symbols = [f"SYM{i:02d}" for i in range(50)]
        added, skipped = await db.add_symbols(user_hash, symbols)
        assert added == 50
        assert skipped == []
        
        # Try to add one more
        added, skipped = await db.add_symbols(user_hash, ["EXTRA"])
        assert added == 0
        assert skipped == ["EXTRA"]
    
    @pytest.mark.asyncio
    async def test_symbols_uppercase(self, db, user_hash):
        """Test that symbols are stored uppercase"""
        await db.add_symbols(user_hash, ["aapl", "msft"])
        symbols = await db.get_watchlist(user_hash)
        assert set(symbols) == {"AAPL", "MSFT"}
    
    @pytest.mark.asyncio
    async def test_user_isolation(self, db):
        """Test that different users have separate watchlists"""
        user1 = hash_phone("+15551111111")
        user2 = hash_phone("+15552222222")
        
        await db.add_symbols(user1, ["AAPL"])
        await db.add_symbols(user2, ["MSFT"])
        
        assert await db.get_watchlist(user1) == ["AAPL"]
        assert await db.get_watchlist(user2) == ["MSFT"]
