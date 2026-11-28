"""Capability tests: mock values are only used in explicitly labelled unit tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from modelforge.platforms.capabilities import (ProbeConfig, classify, measure_capabilities,
                                               memory_budget, _model_spec, _param_count)


def test_mock_budget_and_classification():
    # Synthetic values: test only, never reported as measurements.
    assert memory_budget(1000, 0.35) == 650
    with pytest.raises(ValueError):
        memory_budget(1, 1.0)
    config = ProbeConfig(candidates=(1000,))
    rows = [{'status':'passed','parameters':900,'weights_changed':True,'step_seconds':0.2},
            {'status':'memory_exceeded','parameters':9000,'weights_changed':True,'step_seconds':0.3}]
    result = classify(rows, config)
    assert result['max_params'] == 900
    assert result['training_from_scratch']['estimated_seconds_per_epoch'] == pytest.approx(20)
    assert classify([{'status':'passed','parameters':123,'weights_changed':False}], config)['max_params'] is None


def test_mock_insufficient_memory_rejects_candidate(monkeypatch):
    import modelforge.platforms.capabilities as caps
    # Simulate an available reference, then insufficient memory for the proposed model.
    monkeypatch.setattr(caps, 'device_info', lambda path: {'platform':'linux','memory':{'available_for_budget_bytes':1000,'available_system_bytes':1000}})
    calls = []
    def fake_worker(payload, deadline, rss_budget):
        calls.append(payload['kind'])
        return {'status':'ok', 'weights_changed':True}
    monkeypatch.setattr(caps, '_worker', fake_worker)
    report = caps.measure_capabilities(ProbeConfig(candidates=(500000,)))
    assert report['candidates'][0]['status'] == 'rejected_estimate_over_budget'
    assert calls == ['matmul', 'step']
    assert report['classification']['max_params'] is None


def test_real_probe_changes_weights_and_reports_measured_json():
    config = ProbeConfig(candidates=(500_000,), total_timeout_seconds=30)
    report = measure_capabilities(config)
    assert json.loads(json.dumps(report))['device']['memory']['available_system_bytes'] > 0
    assert report['benchmark']['matmul']['gflops'] > 0
    assert report['benchmark']['reference_step']['weights_changed'] is True
    assert report['benchmark']['reference_step']['peak_rss_bytes'] > 0
    assert report['candidates'][0]['status'] == 'passed'
    assert report['candidates'][0]['weights_changed'] is True
    assert report['classification']['max_params'] == report['candidates'][0]['parameters']


def test_model_size_formula_matches_real_model():
    from modelforge.architectures.models import build_model_from_spec
    spec = _model_spec(500_000, ProbeConfig())
    assert _param_count(spec) == build_model_from_spec(spec).parameter_count()


def test_api_and_cli_expose_same_service(tmp_path: Path, monkeypatch, capsys):
    from modelforge.core.runtime import ModelForgeCore
    from modelforge.backend.api import create_app
    from modelforge.cli.main import main
    # Mock avoids running expensive real probes twice; the integration test above is real.
    sentinel = {'schema_version':1, 'source':'mock-test-only'}
    monkeypatch.setattr(ModelForgeCore, 'capabilities', lambda self: sentinel)
    core = ModelForgeCore(home_dir=tmp_path/'api')
    try:
        with TestClient(create_app(core=core)) as client:
            assert client.get('/api/v1/system/capabilities').json() == sentinel
    finally:
        core.shutdown()
    output = tmp_path/'report.json'
    assert main(['--home', str(tmp_path/'cli'), 'doctor', '--capabilities', '--output', str(output)]) == 0
    assert json.loads(capsys.readouterr().out) == sentinel
    assert json.loads(output.read_text()) == sentinel
