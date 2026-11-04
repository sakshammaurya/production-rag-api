import hmac
from dataclasses import dataclass

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.config import get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@dataclass(frozen=True)
class TenantContext:
    tenant_id: str


def authenticate_tenant(api_key: str | None = Security(api_key_header)) -> TenantContext:
    """Resolve a tenant from a configured API key without trusting caller-supplied identity."""
    configured_keys = get_settings().tenant_api_keys
    if not configured_keys:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured",
        )
    if api_key:
        for tenant_id, expected_key in configured_keys.items():
            if hmac.compare_digest(api_key, expected_key.get_secret_value()):
                return TenantContext(tenant_id=tenant_id)
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key",
    )
