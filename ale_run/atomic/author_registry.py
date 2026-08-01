"""Idempotent private registry for ready authored evaluators."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import secrets
import stat
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from .contracts import (
    AtomicInfrastructureError,
    AuthorEvaluatorRegistryIdentity,
    EvaluatorRegistryRecord,
)
from .evaluator_registry import evaluator_registry_key

_MAX_RECORD_BYTES = 1024 * 1024
_PRIVATE_DIRECTORY_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600
_CLAIM_POLL_SECONDS = 0.01
_IDENTITY_FIELDS = (
    "task_commit",
    "rubric_hash",
    "reference_manifest_hash",
    "evaluator_sdk_version",
    "image_id",
)


class AuthorRegistryTransaction(Protocol):
    """One same-identity authoring transaction."""

    def get_ready(self) -> EvaluatorRegistryRecord | None: ...

    def publish_ready(self, record: EvaluatorRegistryRecord) -> EvaluatorRegistryRecord: ...


class AuthorRegistry(Protocol):
    """Injected persistence boundary for authored evaluator identities."""

    def claim(
        self,
        identity: AuthorEvaluatorRegistryIdentity,
    ) -> AbstractAsyncContextManager[AuthorRegistryTransaction]: ...

    def get_ready(
        self,
        identity: AuthorEvaluatorRegistryIdentity,
    ) -> EvaluatorRegistryRecord | None: ...

    def publish_ready(
        self,
        identity: AuthorEvaluatorRegistryIdentity,
        record: EvaluatorRegistryRecord,
    ) -> EvaluatorRegistryRecord: ...


def author_registry_key(identity: AuthorEvaluatorRegistryIdentity) -> str:
    """Return the canonical SHA-256 key for the five-part authoring identity."""
    encoded = json.dumps(
        {field: getattr(identity, field) for field in _IDENTITY_FIELDS},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class FilesystemAuthorRegistry:
    """Filesystem registry with process-safe same-key serialization."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._claim_locks: dict[str, threading.Lock] = {}
        self._claim_locks_guard = threading.Lock()
        if not self.root.is_absolute():
            raise AtomicInfrastructureError(
                "author_registry",
                "author registry root must be absolute",
            )
        try:
            self.root.mkdir(mode=_PRIVATE_DIRECTORY_MODE)
        except FileExistsError:
            pass
        except OSError as exc:
            raise AtomicInfrastructureError(
                "author_registry",
                f"cannot create author registry root: {exc}",
            ) from exc
        root_fd = self._open_root()
        os.close(root_fd)

    @asynccontextmanager
    async def claim(
        self,
        identity: AuthorEvaluatorRegistryIdentity,
    ) -> AsyncIterator[AuthorRegistryTransaction]:
        """Hold the same-identity lock without blocking the event loop."""
        key = author_registry_key(identity)
        local_lock = self._claim_lock(key)
        local_lock_acquired = False
        root_fd: int | None = None
        lock_fd: int | None = None
        try:
            while not local_lock.acquire(blocking=False):
                await asyncio.sleep(_CLAIM_POLL_SECONDS)
            local_lock_acquired = True
            root_fd = self._open_root()
            lock_fd = self._open_lock(root_fd, key)
            while True:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    await asyncio.sleep(_CLAIM_POLL_SECONDS)
            yield _FilesystemAuthorRegistryTransaction(self, identity, root_fd, key)
        except AtomicInfrastructureError:
            raise
        except OSError as exc:
            raise AtomicInfrastructureError(
                "author_registry",
                f"cannot claim author registry transaction: {exc}",
            ) from exc
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
            if root_fd is not None:
                os.close(root_fd)
            if local_lock_acquired:
                local_lock.release()

    def get_ready(
        self,
        identity: AuthorEvaluatorRegistryIdentity,
    ) -> EvaluatorRegistryRecord | None:
        key = author_registry_key(identity)
        try:
            with self._locked(key) as root_fd:
                return self._read_record(root_fd, key, identity)
        except AtomicInfrastructureError:
            raise
        except OSError as exc:
            raise AtomicInfrastructureError(
                "author_registry",
                f"cannot read author registry: {exc}",
            ) from exc

    def publish_ready(
        self,
        identity: AuthorEvaluatorRegistryIdentity,
        record: EvaluatorRegistryRecord,
    ) -> EvaluatorRegistryRecord:
        self._validate_identity(identity, record)
        key = author_registry_key(identity)
        try:
            with self._locked(key) as root_fd:
                return self._publish_ready_locked(root_fd, key, identity, record)
        except AtomicInfrastructureError:
            raise
        except OSError as exc:
            raise AtomicInfrastructureError(
                "author_registry",
                f"cannot publish author registry record: {exc}",
            ) from exc

    @contextmanager
    def _locked(self, key: str) -> Iterator[int]:
        local_lock = self._claim_lock(key)
        local_lock.acquire()
        root_fd: int | None = None
        lock_fd: int | None = None
        try:
            root_fd = self._open_root()
            lock_fd = self._open_lock(root_fd, key)
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            yield root_fd
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
            if root_fd is not None:
                os.close(root_fd)
            local_lock.release()

    def _claim_lock(self, key: str) -> threading.Lock:
        with self._claim_locks_guard:
            return self._claim_locks.setdefault(key, threading.Lock())

    def _publish_ready_locked(
        self,
        root_fd: int,
        key: str,
        identity: AuthorEvaluatorRegistryIdentity,
        record: EvaluatorRegistryRecord,
    ) -> EvaluatorRegistryRecord:
        self._validate_identity(identity, record)
        try:
            existing = self._read_record(root_fd, key, identity)
            if existing is not None:
                if existing.model_copy(update={"ready_at": record.ready_at}) != record:
                    raise AtomicInfrastructureError(
                        "author_registry",
                        "conflicting ready record exists for authoring identity",
                    )
                self._publish_evaluator_index(identity, existing, key)
                return existing
            self._write_record(root_fd, key, record)
            self._publish_evaluator_index(identity, record, key)
            return record
        except AtomicInfrastructureError:
            raise
        except OSError as exc:
            raise AtomicInfrastructureError(
                "author_registry",
                f"cannot publish author registry record: {exc}",
            ) from exc

    def _publish_evaluator_index(
        self,
        identity: AuthorEvaluatorRegistryIdentity,
        record: EvaluatorRegistryRecord,
        author_key: str,
    ) -> None:
        evaluator_key = evaluator_registry_key(record)
        if evaluator_key == author_key:
            return
        with self._locked(evaluator_key) as root_fd:
            existing = self._read_record(root_fd, evaluator_key, identity)
            if existing is not None:
                if existing != record:
                    raise AtomicInfrastructureError(
                        "author_registry",
                        "conflicting ready record exists for evaluator identity",
                    )
                return
            self._write_record(root_fd, evaluator_key, record)

    def _write_record(
        self,
        root_fd: int,
        key: str,
        record: EvaluatorRegistryRecord,
    ) -> None:
        payload = record.model_dump_json().encode("utf-8")
        if len(payload) > _MAX_RECORD_BYTES:
            raise AtomicInfrastructureError(
                "author_registry",
                "author registry record exceeds the 1 MiB limit",
            )
        temporary_name = f".{key}.{secrets.token_hex(16)}.tmp"
        try:
            temporary_fd = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                _PRIVATE_FILE_MODE,
                dir_fd=root_fd,
            )
            try:
                with os.fdopen(temporary_fd, "wb") as stream:
                    temporary_fd = -1
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            finally:
                if temporary_fd != -1:
                    os.close(temporary_fd)
            os.replace(
                temporary_name,
                f"{key}.json",
                src_dir_fd=root_fd,
                dst_dir_fd=root_fd,
            )
            temporary_name = ""
            os.fsync(root_fd)
        finally:
            if temporary_name:
                try:
                    os.unlink(temporary_name, dir_fd=root_fd)
                except FileNotFoundError:
                    pass

    def _open_root(self) -> int:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        current_fd = os.open("/", flags)
        try:
            for component in self.root.parts[1:]:
                if component in {"", ".", ".."}:
                    raise AtomicInfrastructureError(
                        "author_registry",
                        "author registry root contains an unsafe path component",
                    )
                status_result = os.stat(
                    component,
                    dir_fd=current_fd,
                    follow_symlinks=False,
                )
                if stat.S_ISLNK(status_result.st_mode):
                    raise AtomicInfrastructureError(
                        "author_registry",
                        "author registry root must not contain symlinks",
                    )
                if not stat.S_ISDIR(status_result.st_mode):
                    raise AtomicInfrastructureError(
                        "author_registry",
                        "author registry root is not a directory",
                    )
                next_fd = os.open(component, flags, dir_fd=current_fd)
                os.close(current_fd)
                current_fd = next_fd
            root_status = os.fstat(current_fd)
            if root_status.st_uid != os.geteuid():
                raise AtomicInfrastructureError(
                    "author_registry",
                    "author registry root has the wrong owner",
                )
            if stat.S_IMODE(root_status.st_mode) != _PRIVATE_DIRECTORY_MODE:
                raise AtomicInfrastructureError(
                    "author_registry",
                    "author registry root permissions must be 0700",
                )
            return current_fd
        except BaseException:
            os.close(current_fd)
            raise

    def _open_lock(self, root_fd: int, key: str) -> int:
        lock_fd = os.open(
            f"{key}.lock",
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
            _PRIVATE_FILE_MODE,
            dir_fd=root_fd,
        )
        try:
            self._validate_private_file(os.fstat(lock_fd), "lock")
            return lock_fd
        except BaseException:
            os.close(lock_fd)
            raise

    def _read_record(
        self,
        root_fd: int,
        key: str,
        identity: AuthorEvaluatorRegistryIdentity,
    ) -> EvaluatorRegistryRecord | None:
        try:
            record_fd = os.open(
                f"{key}.json",
                os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW,
                dir_fd=root_fd,
            )
        except FileNotFoundError:
            return None
        try:
            self._validate_private_file(os.fstat(record_fd), "record")
            with os.fdopen(record_fd, "rb") as stream:
                record_fd = -1
                payload = stream.read(_MAX_RECORD_BYTES + 1)
        finally:
            if record_fd != -1:
                os.close(record_fd)
        if len(payload) > _MAX_RECORD_BYTES:
            raise AtomicInfrastructureError(
                "author_registry",
                "author registry record exceeds the 1 MiB limit",
            )
        try:
            record = EvaluatorRegistryRecord.model_validate_json(payload)
        except ValidationError as exc:
            raise AtomicInfrastructureError(
                "author_registry",
                f"invalid author registry record: {exc}",
            ) from exc
        self._validate_identity(identity, record)
        return record

    @staticmethod
    def _validate_private_file(status_result: os.stat_result, label: str) -> None:
        if not stat.S_ISREG(status_result.st_mode):
            raise AtomicInfrastructureError(
                "author_registry",
                f"author registry {label} is not a regular file",
            )
        if status_result.st_uid != os.geteuid():
            raise AtomicInfrastructureError(
                "author_registry",
                f"author registry {label} has the wrong owner",
            )
        if stat.S_IMODE(status_result.st_mode) != _PRIVATE_FILE_MODE:
            raise AtomicInfrastructureError(
                "author_registry",
                f"author registry {label} permissions must be 0600",
            )
        if status_result.st_nlink != 1:
            raise AtomicInfrastructureError(
                "author_registry",
                f"author registry {label} must have one link",
            )

    @staticmethod
    def _validate_identity(
        identity: AuthorEvaluatorRegistryIdentity,
        record: EvaluatorRegistryRecord,
    ) -> None:
        mismatches = [
            field
            for field in _IDENTITY_FIELDS
            if getattr(identity, field) != getattr(record, field)
        ]
        if mismatches:
            raise AtomicInfrastructureError(
                "author_registry",
                f"author registry identity mismatch: {', '.join(mismatches)}",
            )


class _FilesystemAuthorRegistryTransaction:
    def __init__(
        self,
        registry: FilesystemAuthorRegistry,
        identity: AuthorEvaluatorRegistryIdentity,
        root_fd: int,
        key: str,
    ) -> None:
        self._registry = registry
        self._identity = identity
        self._root_fd = root_fd
        self._key = key

    def get_ready(self) -> EvaluatorRegistryRecord | None:
        try:
            return self._registry._read_record(self._root_fd, self._key, self._identity)
        except AtomicInfrastructureError:
            raise
        except OSError as exc:
            raise AtomicInfrastructureError(
                "author_registry",
                f"cannot read author registry: {exc}",
            ) from exc

    def publish_ready(self, record: EvaluatorRegistryRecord) -> EvaluatorRegistryRecord:
        return self._registry._publish_ready_locked(
            self._root_fd,
            self._key,
            self._identity,
            record,
        )
