"""
Admin password setup for the admin web UI.

    python -m src.admin_setup

Writes the bcrypt password hash to `data/admin_password.hash` (mode 0600)
and prints a fresh FLASK_SECRET_KEY to paste into `.env`.

The hash is stored in a file rather than an env var so '$' characters in
the bcrypt format survive docker-compose's variable interpolation.
"""

import getpass
import os
import secrets
import sys
from pathlib import Path

import bcrypt


HASH_PATH = Path(os.getenv("ADMIN_PASSWORD_HASH_PATH", "data/admin_password.hash"))


def main() -> int:
    print("Signal Stock Bot — admin password setup")
    print()

    while True:
        password = getpass.getpass("New admin password: ")
        if len(password) < 12:
            print("Too short — use at least 12 characters.\n", file=sys.stderr)
            continue
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            print("Passwords do not match.\n", file=sys.stderr)
            continue
        break

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()

    HASH_PATH.parent.mkdir(parents=True, exist_ok=True)
    HASH_PATH.write_text(hashed + "\n")
    os.chmod(HASH_PATH, 0o600)

    print()
    print(f"✓ Password hash written to {HASH_PATH} (mode 0600)")

    if not os.getenv("FLASK_SECRET_KEY"):
        secret = secrets.token_urlsafe(48)
        print()
        print("No FLASK_SECRET_KEY detected. Add this to your .env:")
        print()
        print(f"FLASK_SECRET_KEY={secret}")
        print()

    print("Remove any existing ADMIN_PASSWORD_HASH=... line from .env — the")
    print("file at data/admin_password.hash is now the source of truth.")
    print()
    print("Restart the bot: docker compose up -d")
    return 0


if __name__ == "__main__":
    sys.exit(main())
