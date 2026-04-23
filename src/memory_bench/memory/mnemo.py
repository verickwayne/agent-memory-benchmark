import asyncio
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from ..models import Document
from .base import MemoryProvider


def _ensure_mnemo_importable() -> None:
    project_path = Path(
        os.environ.get("MNEMO_PROJECT_PATH", "/Users/verickwayne/Projects/agent-memory")
    ).expanduser()
    if project_path.exists():
        path_str = str(project_path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _format_doc(doc: Document) -> str:
    lines: list[str] = []
    if doc.timestamp:
        lines.append(f"Timestamp: {doc.timestamp}")
    if doc.context:
        lines.append(f"Context: {doc.context}")
    lines.append(f"Document ID: {doc.id}")
    lines.append("")
    lines.append(doc.content)
    return "\n".join(lines).replace("\x00", "")


class MnemoMemoryProvider(MemoryProvider):
    name = "mnemo"
    description = "Local Mnemo SQLite provider using the same recall stack as the CLI/MCP server."
    kind = "local"
    link = "https://github.com/verickwayne/agent-memory"
    concurrency = 1

    def __init__(self):
        self._db_path: Path | None = None
        self._base_group = "amb"
        self._clients: dict[str, object] = {}

    def prepare(
        self,
        store_dir: Path,
        unit_ids: set[str] | None = None,
        reset: bool = True,
    ) -> None:
        if reset and store_dir.exists():
            shutil.rmtree(store_dir)
        store_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = store_dir / "mnemo.db"
        self._base_group = "amb"

        # Keep benchmark imports from producing shadow-git memory logs.
        os.environ.setdefault("MNEMO_MEMORY_LOG", "0")
        _ensure_mnemo_importable()

    def _group_for(self, user_id: str | None) -> str:
        return f"{self._base_group}:{user_id}" if user_id else self._base_group

    def _client(self, group_id: str):
        if self._db_path is None:
            raise RuntimeError("Mnemo provider has not been prepared")
        client = self._clients.get(group_id)
        if client is None:
            from mnemo.client import Mnemo

            client = Mnemo(
                backend="sqlite",
                db_path=str(self._db_path),
                group_id=group_id,
            )
            # Benchmark ingestion is explicit; avoid a background consolidator.
            client._consolidator_disabled = True
            client._memory_log_disabled = True
            self._clients[group_id] = client
        return client

    async def _ingest_async(self, documents: list[Document]) -> None:
        for doc in documents:
            group_id = self._group_for(doc.user_id)
            await self._client(group_id).retain(
                _format_doc(doc),
                source="amb",
                source_description=f"AMB document {doc.id}",
                group_id=group_id,
                dedupe_window_hours=0,
                importance=0.5,
                review_state="approved",
                created_at=_parse_datetime(doc.timestamp),
                enqueue_consolidation=False,
            )

    def ingest(self, documents: list[Document]) -> None:
        asyncio.run(self._ingest_async(documents))

    async def async_ingest(self, documents: list[Document]) -> None:
        await self._ingest_async(documents)

    async def async_retrieve(
        self,
        query: str,
        k: int = 10,
        user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[list[Document], dict | None]:
        group_id = self._group_for(user_id)
        results = await self._client(group_id).recall(
            query,
            limit=k,
            reference_date=_parse_datetime(query_timestamp),
        )
        docs = [
            Document(
                id=(r.episode_uuids[0] if r.episode_uuids else f"mnemo:{i}"),
                content=r.fact,
                user_id=user_id,
            )
            for i, r in enumerate(results)
        ]
        raw = {"results": [r.to_dict() for r in results]}
        return docs, raw

    def retrieve(
        self,
        query: str,
        k: int = 10,
        user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[list[Document], dict | None]:
        return asyncio.run(
            self.async_retrieve(
                query,
                k=k,
                user_id=user_id,
                query_timestamp=query_timestamp,
            )
        )

    async def _close_all(self) -> None:
        for client in list(self._clients.values()):
            await client.close()
        self._clients.clear()

    def cleanup(self) -> None:
        if not self._clients:
            return
        asyncio.run(self._close_all())
