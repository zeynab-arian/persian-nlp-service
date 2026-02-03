from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request

class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request.state.person_id = request.headers.get("PERSON-ID")
        request.state.remote_address = request.headers.get("X-Forwarded-For", request.client.host)
        response = await call_next(request)
        return response