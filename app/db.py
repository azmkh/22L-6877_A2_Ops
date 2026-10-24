"""Task 5 - the prediction trail. One row per answered request: what was decided, when, by which
model (name + registry version), how confidently, and where inference ran."""
import logging
import time
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, create_engine, func, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

log = logging.getLogger("visionguard.db")


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True,
                                                 default=lambda: datetime.now(timezone.utc))
    filename: Mapped[str] = mapped_column(String(255))
    file_sha256: Mapped[str] = mapped_column(String(64))
    file_size: Mapped[int] = mapped_column(Integer)
    label: Mapped[str] = mapped_column(String(16), index=True)
    probability_ai: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(128))
    model_version: Mapped[str] = mapped_column(String(64))
    inference_source: Mapped[str] = mapped_column(String(255))
    latency_ms: Mapped[float] = mapped_column(Float)


class Database:
    def __init__(self, url: str):
        kwargs = {"pool_pre_ping": True}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
        self.url = url
        self.engine = create_engine(url, **kwargs)
        self.Session = sessionmaker(self.engine, expire_on_commit=False)

    def init(self, retries: int = 5, delay: float = 2.0):
        """Create the table. Retries briefly so a database that is still starting is tolerated,
        but a real misconfiguration (e.g. wrong password) fails loudly in the logs."""
        safe_url = self.engine.url.render_as_string(hide_password=True)
        for attempt in range(1, retries + 1):
            try:
                Base.metadata.create_all(self.engine)
                log.info("Database ready at %s", safe_url)
                return
            except Exception as exc:
                log.error("Database connection failed (attempt %d/%d) at %s: %s",
                          attempt, retries, safe_url, exc)
                if attempt == retries:
                    raise
                time.sleep(delay)

    def ping(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as exc:
            log.error("Database health check failed: %s", exc)
            return False

    def add(self, **fields) -> Prediction:
        with self.Session.begin() as session:
            row = Prediction(**fields)
            session.add(row)
        return row

    def get(self, prediction_id: int):
        with self.Session() as session:
            return session.get(Prediction, prediction_id)

    def recent(self, limit: int = 200, label=None):
        query = select(Prediction).order_by(Prediction.created_at.desc(), Prediction.id.desc()).limit(limit)
        if label:
            query = query.where(Prediction.label == label)
        with self.Session() as session:
            return list(session.scalars(query))

    def counts(self) -> dict:
        query = select(Prediction.label, func.count()).group_by(Prediction.label)
        with self.Session() as session:
            return dict(session.execute(query).all())
