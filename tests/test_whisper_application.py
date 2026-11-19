"""Tests for the WhisperApplication entry point."""

from __future__ import annotations

import os

import pytest
from roslibpy.core import RosTimeoutError

from src.openduck_parts.main import (
    ApplicationConfig,
    WhisperApplication,
    build_config,
)


def make_config(**overrides) -> ApplicationConfig:
    values = {
        "host": "localhost",
        "port": 9090,
        "model_name": None,
        "language": None,
        "device": None,
        "compute_type": None,
        "pcm_queue_size": 150,
        "segment_queue_size": 10,
        "drain_on_shutdown": True,
    }
    values.update(overrides)
    return ApplicationConfig(**values)


class FakeRos:
    def __init__(
        self,
        host="localhost",
        port=9090,
        connected=True,
        run_error=None,
    ):
        self.host = host
        self.port = port
        self.is_connected = connected
        self.run_error = run_error
        self.run_calls = []
        self.terminate_count = 0

    def run(self, timeout=10):
        self.run_calls.append(timeout)
        if self.run_error is not None:
            raise self.run_error

    def terminate(self):
        self.terminate_count += 1


class FakeService:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.start_count = 0
        self.stop_calls = []

    def start(self):
        self.start_count += 1

    def stop(self, *, drain=True):
        self.stop_calls.append(drain)


def test_ros_factory_receives_host_port_and_run():
    ros_calls = []

    def ros_factory(*, host, port):
        ros_calls.append((host, port))
        ros = FakeRos(host, port)
        ros.is_connected = False
        return ros

    application = WhisperApplication(
        make_config(host="robot", port=9191),
        ros_factory=ros_factory,
        service_factory=FakeService,
    )

    assert application.run() == 0
    assert ros_calls == [("robot", 9191)]


def test_run_and_service_share_same_ros_client():
    ros = FakeRos()
    ros.is_connected = False

    def service_factory(**kwargs):
        assert kwargs["ros_client"] is ros
        service = FakeService(**kwargs)
        service.start = lambda: setattr(ros, "is_connected", False)
        return service

    application = WhisperApplication(
        make_config(),
        ros_client=ros,
        service_factory=service_factory,
    )

    assert application.run() == 0
    assert ros.run_calls == [10]


def test_request_stop_ends_wait_loop():
    ros = FakeRos()
    service = FakeService
    application = WhisperApplication(
        make_config(),
        ros_client=ros,
        service_factory=lambda **kwargs: service(**kwargs),
    )
    application.request_stop()

    assert application.run() == 0


def test_cleanup_stops_service_before_terminate():
    ros = FakeRos()
    order = []

    class OrderedService(FakeService):
        def start(self):
            self.start_count += 1

        def stop(self, *, drain=True):
            order.append(("service_stop", drain))

    def terminate():
        order.append("ros_terminate")

    ros.terminate = terminate
    ros.is_connected = False
    application = WhisperApplication(
        make_config(drain_on_shutdown=False),
        ros_client=ros,
        service_factory=lambda **kwargs: OrderedService(**kwargs),
    )

    application.run()

    assert order == [
        ("service_stop", False),
        "ros_terminate",
    ]


def test_cleanup_twice_is_safe():
    ros = FakeRos()
    ros.is_connected = False
    application = WhisperApplication(
        make_config(),
        ros_client=ros,
        service_factory=FakeService,
    )

    application.run()
    application.cleanup()

    assert ros.terminate_count == 1


def test_service_stop_failure_still_terminates_ros():
    ros = FakeRos()

    class FailingService(FakeService):
        def stop(self, *, drain=True):
            raise RuntimeError("stop failed")

    ros.is_connected = False
    application = WhisperApplication(
        make_config(),
        ros_client=ros,
        service_factory=lambda **kwargs: FailingService(**kwargs),
    )

    assert application.run() == 0
    assert ros.terminate_count == 1


def test_service_start_failure_returns_one_and_cleanup():
    ros = FakeRos()
    ros.is_connected = False

    class FailingService(FakeService):
        def start(self):
            raise RuntimeError("start failed")

    application = WhisperApplication(
        make_config(),
        ros_client=ros,
        service_factory=lambda **kwargs: FailingService(**kwargs),
    )

    assert application.run() == 1
    assert ros.terminate_count == 1


def test_ros_timeout_returns_one():
    ros = FakeRos(run_error=RosTimeoutError("timeout"))
    application = WhisperApplication(
        make_config(),
        ros_client=ros,
    )

    assert application.run() == 1
    assert ros.terminate_count == 1


def test_no_drain_is_reflected_in_config():
    config = build_config([
        "--no-drain",
        "--host",
        "robot",
        "--port",
        "9191",
    ])

    assert config.drain_on_shutdown is False
    assert config.host == "robot"
    assert config.port == 9191


def test_environment_defaults_are_used(monkeypatch):
    monkeypatch.setenv("ROSBRIDGE_HOST", "env-host")
    monkeypatch.setenv("ROSBRIDGE_PORT", "9192")

    config = build_config([])

    assert config.host == "env-host"
    assert config.port == 9192


def test_cli_overrides_environment(monkeypatch):
    monkeypatch.setenv("ROSBRIDGE_HOST", "env-host")
    monkeypatch.setenv("ROSBRIDGE_PORT", "9192")

    config = build_config([
        "--host",
        "cli-host",
        "--port",
        "9193",
    ])

    assert config.host == "cli-host"
    assert config.port == 9193


def test_cli_options_are_reflected_in_config():
    config = build_config([
        "--model",
        "small",
        "--language",
        "ja",
        "--device",
        "cuda",
        "--compute-type",
        "float16",
        "--pcm-queue-size",
        "200",
        "--segment-queue-size",
        "12",
    ])

    assert config.model_name == "small"
    assert config.language == "ja"
    assert config.device == "cuda"
    assert config.compute_type == "float16"
    assert config.pcm_queue_size == 200
    assert config.segment_queue_size == 12


def test_request_stop_is_idempotent():
    application = WhisperApplication(make_config())
    application.request_stop()
    application.request_stop()

    assert application._stop_event.is_set()
