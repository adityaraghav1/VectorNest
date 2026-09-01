"""Centralized HTTP error handling for the VectorNest API."""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from vectornest.core.exceptions import (
    CollectionNotFoundError,
    DuplicateCollectionError,
    DuplicateRecordError,
    RecordNotFoundError,
    ValidationError,
)


async def not_found_handler(
    _request: Request,
    exc: Exception,
) -> JSONResponse:
    """Convert missing-resource domain errors into HTTP 404."""

    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": str(exc)},
    )


async def conflict_handler(
    _request: Request,
    exc: Exception,
) -> JSONResponse:
    """Convert duplicate-resource errors into HTTP 409."""

    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={"detail": str(exc)},
    )


async def validation_handler(
    _request: Request,
    exc: Exception,
) -> JSONResponse:
    """Convert VectorNest validation errors into HTTP 422."""

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": str(exc)},
    )


def register_exception_handlers(
    application: FastAPI,
) -> None:
    """Register VectorNest domain exception handlers."""

    application.add_exception_handler(
        CollectionNotFoundError,
        not_found_handler,
    )
    application.add_exception_handler(
        RecordNotFoundError,
        not_found_handler,
    )

    application.add_exception_handler(
        DuplicateCollectionError,
        conflict_handler,
    )
    application.add_exception_handler(
        DuplicateRecordError,
        conflict_handler,
    )

    application.add_exception_handler(
        ValidationError,
        validation_handler,
    )