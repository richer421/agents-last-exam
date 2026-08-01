from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from ale_run.base_interface import SandboxSpec
from ale_run.environments.providers import aliyun as aliyun_module
from ale_run.environments.providers.aliyun import AliyunProvider
from ale_run.orchestration.config_loader import load_experiment


def _provider_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "region": "ap-southeast-1",
        "security_group": "ale-sandbox",
        "snapshots": {
            "cpu-free": {
                "image": "ale-win10",
                "zones": ["ap-southeast-1a", "ap-southeast-1b"],
            }
        },
    }
    config.update(overrides)
    return config


def test_provider_config_uses_requested_cpu_family() -> None:
    provider = AliyunProvider(_provider_config(cpu_instance_family="ecs.g7nex"))

    assert provider.config.cpu_instance_family == "ecs.g7nex"
    assert (
        aliyun_module._resolve_instance_type(
            "c4-standard-4",
            is_gpu=False,
            cpu_instance_family=provider.config.cpu_instance_family,
        )
        == "ecs.g7nex.xlarge"
    )


def test_eof_is_classified_as_transient_transport() -> None:
    assert aliyun_module._is_transient_error(
        'ERROR: Post "https://ecs.example.invalid/?AccessKeyId=secret": EOF'
    )


def test_aliyun_error_sanitizer_redacts_signed_query_values() -> None:
    sanitized = aliyun_module._sanitize_aliyun_error(
        "Post https://ecs.example.invalid/?AccessKeyId=abc&Signature=xyz: EOF"
    )
    assert "abc" not in sanitized
    assert "xyz" not in sanitized
    assert "AccessKeyId=[REDACTED]" in sanitized
    assert "Signature=[REDACTED]" in sanitized


@pytest.mark.asyncio
async def test_aliyun_wrapper_redacts_signed_query_values_from_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        returncode = 1

        async def communicate(self) -> tuple[bytes, bytes]:
            return (
                b"Post https://ecs.invalid/?AccessKeyId=abc&Signature=xyz: EOF",
                b"",
            )

    async def fake_subprocess(*args: object, **kwargs: object) -> Process:
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subprocess)

    rc, stdout, stderr = await aliyun_module._run_aliyun("ecs", "DescribeInstances")

    assert rc == 1
    assert stderr == ""
    assert "abc" not in stdout
    assert "xyz" not in stdout
    assert "AccessKeyId=[REDACTED]" in stdout


@pytest.mark.asyncio
async def test_security_group_resolution_retries_transient_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = AliyunProvider(_provider_config())
    calls = 0

    async def fake_run(*args: str) -> tuple[int, str, str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return 1, "", "Post https://ecs.example.invalid/: EOF"
        return 0, json.dumps(
            {
                "SecurityGroups": {
                    "SecurityGroup": [
                        {"SecurityGroupId": "sg-test", "VpcId": "vpc-test"}
                    ]
                }
            }
        ), ""

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(aliyun_module, "_run_aliyun", fake_run)
    monkeypatch.setattr(aliyun_module.asyncio, "sleep", no_sleep)

    assert await provider._resolve_security_group() == ("sg-test", "vpc-test")
    assert calls == 2


def test_loader_keeps_aliyun_cpu_instance_family(tmp_path: Path) -> None:
    agent = tmp_path / "agent.yaml"
    agent.write_text("harness: dummy\nmodel: test\n", encoding="utf-8")
    environment = tmp_path / "environment.yaml"
    environment.write_text(
        """
snapshots:
  cpu-free:
    provider: aliyun
    image: ale-win10
    resolution: [1024, 768]
    aliyun:
      region: ap-southeast-1
      security_group: ale-sandbox
      cpu_instance_family: ecs.g7nex
      zones: [ap-southeast-1a]
task_data_source: baked_in_sandbox
output_path: null
""",
        encoding="utf-8",
    )
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        f"""
name: aliyun-test
agent: {agent}
environment: {environment}
tasks:
  - path: demo/hello_win
""",
        encoding="utf-8",
    )

    spec = load_experiment(experiment)
    config = spec.environment.provider_specs["aliyun"].config

    assert config["cpu_instance_family"] == "ecs.g7nex"
    assert config["snapshots"]["cpu-free"]["zones"] == ["ap-southeast-1a"]


def test_run_args_attach_preconfigured_security_group() -> None:
    cfg = AliyunProvider(_provider_config()).config

    args = aliyun_module._build_run_args(
        name="ale-demo-1234",
        image_id="m-image",
        instance_type="ecs.g7nex.xlarge",
        zone_id="ap-southeast-1a",
        vswitch_id="vsw-test",
        security_group_id="sg-base",
        cfg=cfg,
        snapshot_tag="cpu-free",
        ram_role_name="",
    )

    assert args[args.index("--SecurityGroupId") + 1] == "sg-base"
    assert not any(arg.startswith("--SecurityGroupIds.") for arg in args)


@pytest.mark.asyncio
async def test_acquire_uses_only_preconfigured_security_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = AliyunProvider(
        _provider_config(cpu_instance_family="ecs.g7nex")
    )
    run_call: dict[str, object] = {}

    async def fake_resolve_image(*args: object) -> str:
        return "m-image"

    async def fake_resolve_security_group() -> tuple[str, str]:
        return "sg-base", "vpc-test"

    async def fake_effective_ram_role() -> str:
        return ""

    async def fake_vswitch_for_zone(*args: object) -> str:
        return "vsw-test"

    async def fake_try_run_in_zone(**kwargs: object) -> tuple[bool, str, str]:
        run_call.update(kwargs)
        return True, json.dumps({"InstanceIdSets": {"InstanceIdSet": ["i-test"]}}), ""

    async def fake_wait_running_with_ip(*args: object, **kwargs: object) -> str:
        return "203.0.113.10"

    async def fake_wait_cua_ready(*args: object, **kwargs: object) -> bool:
        return True

    monkeypatch.setattr(provider, "_resolve_image", fake_resolve_image)
    monkeypatch.setattr(provider, "_resolve_security_group", fake_resolve_security_group)
    monkeypatch.setattr(provider, "_effective_ram_role", fake_effective_ram_role)
    monkeypatch.setattr(provider, "_vswitch_for_zone", fake_vswitch_for_zone)
    monkeypatch.setattr(aliyun_module, "_try_run_in_zone", fake_try_run_in_zone)
    monkeypatch.setattr(aliyun_module, "_wait_running_with_ip", fake_wait_running_with_ip)
    monkeypatch.setattr(aliyun_module, "wait_cua_ready", fake_wait_cua_ready)

    sandbox = await provider.acquire(
        SandboxSpec(snapshot="cpu-free", os="windows", machine_type="c4-standard-4")
    )

    assert sandbox.id == "i-test"
    assert run_call["security_group_id"] == "sg-base"
    assert run_call["instance_type"] == "ecs.g7nex.xlarge"
    assert "cua_security_group_id" not in sandbox.metadata
