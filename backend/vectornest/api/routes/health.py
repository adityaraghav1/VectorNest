"""Health-check endpoint."""

from fastapi import APIRouter

from vectornest.api.schemas.common import HealthResponse

router = APIRouter(
    tags=["health"],
)


@router.get(
    "/health",
    response_model=HealthResponse,
)
def health_check() -> HealthResponse:
    """Return the current VectorNest service status."""

    return HealthResponse(
        status="ok",
        service="vectornest",
    )