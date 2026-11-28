from __future__ import annotations

import numpy as np
import pytest

from modelforge.mobile.offline import generate, train


def test_train_saves_real_weights_and_can_generate(tmp_path):
    text = 'Привет мир! Салом ҷаҳон!\n' * 8
    result = train(str(tmp_path), 'Языковой помощник', text, passes=3, max_seconds=30)
    assert result['steps'] > 0
    assert result['spec']['loop_passes'] == 3
    assert result['weight_mb'] <= result['requested_weight_mb']
    with np.load(tmp_path / 'offline_model.npz', allow_pickle=False) as data:
        assert 'tok_emb' in data.files and 'lm_head' in data.files
    assert isinstance(generate(str(tmp_path), 'Привет', 5), str)


def test_failed_training_keeps_previous_checkpoint(tmp_path):
    train(str(tmp_path), 'test', 'abc abc abc')
    before = (tmp_path / 'offline_model.npz').read_bytes()
    with pytest.raises(ValueError):
        train(str(tmp_path), 'test', ' ', passes=3)
    assert (tmp_path / 'offline_model.npz').read_bytes() == before


@pytest.mark.parametrize('passes', [0, 9, True, 1.5])
def test_bad_passes_rejected(tmp_path, passes):
    with pytest.raises(ValueError):
        train(str(tmp_path), 'test', 'abc', passes=passes)


def test_android_scaffold_patch_is_idempotent(tmp_path, monkeypatch):
    from scripts import prepare_flutter_platforms as prepare
    app = tmp_path / 'android/app'
    app.mkdir(parents=True)
    (tmp_path / 'android/settings.gradle').write_text('plugins {\n    id "com.android.application" version "8"\n}\n')
    (app / 'build.gradle').write_text('plugins {\n    id "kotlin-android"\n}\nandroid { minSdk = flutter.minSdkVersion }\n')
    monkeypatch.setattr(prepare, 'FLUTTER_DIR', tmp_path)
    prepare.configure_android_offline()
    prepare.configure_android_offline()
    build = (app / 'build.gradle').read_text()
    assert build.count('id "com.chaquo.python"') == 1
    assert 'minSdk = 24' in build
    assert (app / 'src/main/python/modelforge/mobile/offline.py').exists()
    assert (app / 'src/main/kotlin/ai/modelforge/modelforge_app/MainActivity.kt').exists()
    assert 'core.runtime' not in (app / 'src/main/python/modelforge/__init__.py').read_text()
