# Source: app/config.py
# Deposited with PulseTech PLOS ONE package (V6)
# No credentials are stored here: every secret defaults to empty and must be set
# in a local .env file (see .env.example), which is never committed.

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://USER:PASSWORD@localhost:5432/faers_pediatric"
    DATABASE_URL_SYNC: str = "postgresql+psycopg2://USER:PASSWORD@localhost:5432/faers_pediatric"
    REDIS_URL: str = "redis://localhost:6379/0"
    RXNORM_API_BASE: str = "https://rxnav.nlm.nih.gov/REST"
    SECRET_KEY: str = ""            # set in .env

    # Pipeline settings
    SKIP_RXNORM: str = "0"
    GNN_CHECKPOINT_DIR: str = "./gnn/checkpoints/"
    GNN_DEVICE: str = "cpu"
    SIGNAL_MIN_N: int = 3
    ACTIVE_QUARTERS: str = ""

    # ── Security settings ───────────────────────────────────────────
    PULSETECH_SECRET_KEY: str = ""  # set in .env
    PULSETECH_ADMIN_USER: str = ""  # set in .env
    PULSETECH_ADMIN_PASS: str = ""  # set in .env
    ALLOWED_ORIGIN: str = "http://localhost:5173"
    PULSETECH_IP_WHITELIST: str = ""
    PULSETECH_RATE_LIMIT: int = 100
    PULSETECH_LICENSE_KEY: str = ""
    PULSETECH_LICENSE_EXPIRY: str = "2027-12-31"
    PULSETECH_INTEGRITY_FILE: str = ""
    PULSETECH_AUDIT_LOG: str = "security_audit.log"
    PULSETECH_PRODUCTION: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
