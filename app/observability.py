import json
import logging

logger = logging.getLogger("uvicorn.error")


def log_event(
    level: int,
    event: str,
    *,
    include_exception: bool = False,
    **fields: object,
) -> None:
    """Emit a JSON event through Uvicorn's configured application logger."""
    payload = json.dumps({"event": event, **fields}, sort_keys=True, separators=(",", ":"))
    logger.log(
        level,
        payload,
        extra=fields,
        exc_info=include_exception,
    )
