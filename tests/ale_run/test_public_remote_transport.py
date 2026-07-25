from __future__ import annotations

import urllib.request
from collections.abc import Callable

import pytest
from computer.interface.generic import GenericComputerInterface

from ale_run.base_interface import SandboxHandle
from ale_run.environments.providers.aliyun import AliyunProvider
from ale_run.environments.providers.gcloud import GcloudProvider


@pytest.fixture(autouse=True)
def _disable_cua_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUA_TELEMETRY_ENABLED", "false")
    monkeypatch.setenv("CUA_TELEMETRY_DISABLED", "true")


def _remote_vm() -> SandboxHandle:
    return SandboxHandle(
        id="remote-vm",
        endpoint="http://203.0.113.10:5000",
        os="windows",
        work_dir_base=r"C:\Users\User\.ale",
        task_data_root=r"C:\agenthle",
        node=r"C:\node.exe",
        python=r"C:\python.exe",
        mcp_server_dir=r"C:\cua_mcp_server",
    )


def test_gcloud_provider_prefers_websocket_when_endpoint_uses_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        urllib.request,
        "getproxies",
        lambda: {"https": "http://user:password@proxy.example:8080"},
    )
    monkeypatch.setattr(urllib.request, "proxy_bypass", lambda host: False)

    session = GcloudProvider.open_session(object(), _remote_vm())

    assert session.interface._send_command.__func__ is GenericComputerInterface._send_command_ws


def test_aliyun_public_provider_keeps_rest_first_when_host_proxy_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        urllib.request,
        "getproxies",
        lambda: {"https": "http://user:password@proxy.example:8080"},
    )
    monkeypatch.setattr(urllib.request, "proxy_bypass", lambda host: False)

    session = AliyunProvider.open_session(object(), _remote_vm())

    assert session.interface._send_command.__func__ is GenericComputerInterface._send_command


@pytest.mark.parametrize("provider_type", [GcloudProvider, AliyunProvider])
@pytest.mark.parametrize(
    ("getproxies", "proxy_bypass"),
    [
        (lambda: {}, lambda host: False),
        (lambda: {"https": "http://proxy.example:8080"}, lambda host: True),
    ],
    ids=["no-proxy", "no-proxy-match"],
)
def test_public_provider_keeps_rest_first_without_applicable_proxy(
    provider_type: type,
    getproxies: Callable[[], dict[str, str]],
    proxy_bypass: Callable[[str], bool],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(urllib.request, "getproxies", getproxies)
    monkeypatch.setattr(urllib.request, "proxy_bypass", proxy_bypass)

    session = provider_type.open_session(object(), _remote_vm())

    assert session.interface._send_command.__func__ is GenericComputerInterface._send_command
