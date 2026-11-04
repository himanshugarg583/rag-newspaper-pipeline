"""Celery application configuration."""

from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "newspaper_rag",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.workers.document_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_track_started=True,
    result_expires=3600,
    broker_connection_retry_on_startup=True,
    task_default_queue="documents",
    # Hard safety limits for long-running PDF processing tasks (e.g. 30-50 page newspaper scans).
    task_time_limit=60 * 60 * 2,
    task_soft_time_limit=60 * 60 * 2 - 300,
)
