# Source: app/database.py
# Deposited with PulseTech PLOS ONE package (V6)
# Credentials and hard-coded paths have been replaced with environment variables.

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from sqlalchemy import create_engine
try:
    from app.config import settings
except ImportError:
    from config import settings

# Async engine (for upload/pipeline operations)
engine = create_async_engine(settings.DATABASE_URL, echo=False, pool_size=5, max_overflow=5, pool_recycle=300)
async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Sync engine (for analytics queries — 100x faster for JOINs)
sync_url = settings.DATABASE_URL.replace("+asyncpg", "+psycopg2").replace("asyncpg://", "psycopg2://")
sync_engine = create_engine(sync_url, pool_size=5, max_overflow=5, pool_recycle=300)
sync_session_factory = sessionmaker(sync_engine, class_=Session, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_sync_db():
    """Sync DB session for analytics queries (much faster for JOINs)."""
    session = sync_session_factory()
    try:
        yield session
    finally:
        session.close()


def get_sync_session():
    """Context-manager sync session for non-FastAPI code (e.g. deduplicator)."""
    return sync_session_factory()

