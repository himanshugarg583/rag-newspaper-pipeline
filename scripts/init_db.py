"""Development helper: create the pgvector extension and run migrations.

Prefer `alembic upgrade head`; this script only exists for convenience when
bootstrapping a fresh development database.

Usage:
    python scripts/init_db.py
"""

import subprocess
import sys

from sqlalchemy import create_engine, text

from app.core.config import settings


def main() -> int:
    engine = create_engine(settings.sync_database_url)
    with engine.connect() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        connection.commit()
    print("pgvector extension ensured.")

    result = subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"])
    if result.returncode != 0:
        print("Alembic migration failed.", file=sys.stderr)
        return result.returncode
    print("Migrations applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
