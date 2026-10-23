import asyncio
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Mapping,
)
from dataclasses import dataclass
from typing import Any

import niquests

from ._json import dumps as _json_dumps, loads as _json_loads
from .errors import LLMError, Provider
from .logs import log

__all__ = ("WsConnection", "aiter_items", "open_ws")


async def aiter_items[T](items: Iterable[T]) -> AsyncIterator[T]:
    """Adapt a sync iterable to the async one the send loops consume."""
    for item in items:
        yield item


async def open_ws(
    session: niquests.AsyncSession,
    url: str,
    *,
    timeout: float,
    provider: Provider,
    params: Any = None,
) -> Any:
    """GET `url` as a WebSocket upgrade and return its extension; a plain HTTP reply raises `LLMError`."""
    resp = await session.get(url, params=params, timeout=timeout)
    if resp.extension is None:
        raise LLMError(
            provider,
            f"handshake did not upgrade to a websocket (status {resp.status_code})",
        )
    return resp.extension


@dataclass
class WsConnection:
    """JSON-over-WebSocket connection: send events, async-iterate parsed messages."""

    ext: Any
    _closed: bool = False

    async def send_event(self, event: Mapping[str, Any]) -> None:
        """Send a raw client event as a JSON text frame."""
        await self.ext.send_payload(_json_dumps(event))

    async def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        while not self._closed:
            try:
                payload = await self.ext.next_payload()
            except niquests.exceptions.ReadTimeout:
                continue
            if payload is None:
                break
            if isinstance(payload, (bytes, bytearray)):
                payload = bytes(payload).decode("utf-8", errors="replace")
            try:
                yield _json_loads(payload)
            except ValueError:
                log.debug("skipping non-JSON ws frame: %.100r", payload)

    async def pump(
        self,
        provider: Provider,
        send_all: Callable[[], Awaitable[None]],
        is_final: Callable[[Mapping[str, Any]], bool],
    ) -> AsyncGenerator[dict[str, Any]]:
        """Run `send_all` in a background task, yielding messages until the final one (not yielded).

        The sender is always cancelled and awaited on exit; a send error is re-raised, and a server
        close before the final message raises `LLMError`.
        """

        async def run() -> None:
            try:
                await send_all()
            except Exception:
                await self.close()  # unblock the reader
                raise

        sender = asyncio.create_task(run())
        final = False
        read_error: Exception | None = None
        try:
            async for message in self:
                if is_final(message):
                    final = True
                    break
                yield message
        except Exception as error:
            read_error = error  # a send failure closes the socket; report the cause
        finally:
            sender.cancel()
            (outcome,) = await asyncio.gather(sender, return_exceptions=True)
        if isinstance(outcome, Exception):
            raise outcome from read_error
        if read_error is not None:
            raise read_error
        if not final:
            raise LLMError(provider, "connection closed before the end of the stream")

    async def wait_for(
        self,
        predicate: Callable[[Mapping[str, Any]], bool],
        *,
        timeout: float,
        provider: Provider,
        what: str,
    ) -> dict[str, Any]:
        """Read until a message satisfies `predicate`; a close or `timeout` raises `LLMError`."""
        try:
            async with asyncio.timeout(timeout):
                async for message in self:
                    if predicate(message):
                        return message
        except TimeoutError as error:
            raise LLMError(provider, f"timed out waiting for {what}") from error
        raise LLMError(provider, f"connection closed before {what}")

    async def close(self) -> None:
        """Close the underlying websocket (idempotent)."""
        if self._closed:
            return
        self._closed = True
        try:
            await self.ext.close()
        except Exception:  # best effort — the socket may already be gone
            log.debug("error closing realtime websocket", exc_info=True)
