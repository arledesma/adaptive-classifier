"""Features for a long-running, continuously updated server.

Cheap atomic saves, calibration bookkeeping, newest-plus-boundary retention, batched OOD
scores and the percentile OOD radius. The HTTP side is in test_serving.py.
"""

import json
from datetime import datetime

import numpy as np
import pytest
import torch

from adaptive_classifier import AdaptiveClassifier
from adaptive_classifier import classifier as classifier_module
from adaptive_classifier.memory import PrototypeMemory
from adaptive_classifier.models import Example, ModelConfig

from .synthetic import Dataset


@pytest.fixture
def data(monkeypatch):
    return Dataset().patch(monkeypatch)


@pytest.fixture
def clf(new_classifier, data):
    texts, labels = data.classes(12, n_classes=2, noise=0.2, seed=1)
    c = new_classifier(head_steps=20)
    c.add_examples(texts, labels)
    return c


def load(path):
    return AdaptiveClassifier.load(str(path), use_onnx=False, device="cpu")


# --- 1. cheap, atomic saves -------------------------------------------------------------------------

@pytest.fixture
def fake_export(monkeypatch):
    """Replace the ONNX export with one that writes a placeholder file and counts calls."""
    calls = []

    def export(self, directory, quantize=False, **kwargs):
        calls.append(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ("model_quantized.onnx" if quantize else "model.onnx")).write_bytes(b"onnx")
        return directory

    monkeypatch.setattr(AdaptiveClassifier, "export_onnx", export)
    return calls


def test_a_second_save_reuses_the_onnx_export_and_tokenizer(clf, tmp_path, fake_export, monkeypatch):
    clf.save(str(tmp_path))
    assert len(fake_export) == 1

    tokenizer_saves = []
    monkeypatch.setattr(clf.tokenizer, "save_pretrained", lambda *a, **k: tokenizer_saves.append(a))
    clf.add_examples(["tr0_0"], ["C0"])
    clf.save(str(tmp_path))
    assert len(fake_export) == 1, "the unchanged encoder was exported again"
    assert tokenizer_saves == []
    assert load(tmp_path).training_history["C0"] == 13          # the learned state was written


def test_onnx_is_exported_again_for_a_different_encoder_or_on_request(clf, tmp_path, fake_export):
    clf.save(str(tmp_path))
    clf.save(str(tmp_path), reuse_onnx=False)
    assert len(fake_export) == 2

    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    config["model_name"] = "some/other-encoder"
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    clf.save(str(tmp_path))
    assert len(fake_export) == 3


def test_saving_without_onnx_never_exports(clf, tmp_path, fake_export):
    clf.save(str(tmp_path), include_onnx=False)
    clf.save(str(tmp_path), include_onnx=False)
    assert fake_export == []


def test_a_failed_save_leaves_the_previous_files_whole(clf, tmp_path, monkeypatch):
    clf.save(str(tmp_path), include_onnx=False)
    before = {name: (tmp_path / name).read_bytes()
              for name in ("config.json", "examples.json", "model.safetensors")}

    clf.add_examples(["tr1_0"], ["C1"])
    real_replace = classifier_module.os.replace
    calls = []

    def failing_replace(src, dst):
        calls.append(dst)
        if len(calls) == 2:                     # the second file of the save
            raise OSError("disk full")
        return real_replace(src, dst)

    monkeypatch.setattr(classifier_module.os, "replace", failing_replace)
    with pytest.raises(OSError, match="disk full"):
        clf.save(str(tmp_path), include_onnx=False)
    monkeypatch.setattr(classifier_module.os, "replace", real_replace)

    assert (tmp_path / "config.json").read_bytes() == before["config.json"]
    assert (tmp_path / "examples.json").read_bytes() == before["examples.json"]
    assert not list(tmp_path.glob("*.tmp")), "a temporary file was left behind"
    json.loads((tmp_path / "examples.json").read_text(encoding="utf-8"))


def test_replacing_a_file_held_open_elsewhere_is_retried(clf, tmp_path, monkeypatch):
    """On Windows os.replace fails while another process has the target open."""
    real_replace = classifier_module.os.replace
    failures = {"left": 2}

    def busy_replace(src, dst):
        if failures["left"]:
            failures["left"] -= 1
            raise PermissionError("in use")
        return real_replace(src, dst)

    monkeypatch.setattr(classifier_module.os, "replace", busy_replace)
    monkeypatch.setattr("time.sleep", lambda _: None)
    clf.save(str(tmp_path), include_onnx=False)
    assert "C0" in load(tmp_path).label_to_id


def test_a_loaded_classifier_can_save_over_its_own_directory(clf, tmp_path):
    clf.save(str(tmp_path), include_onnx=False)
    loaded = load(tmp_path)                         # must not keep model.safetensors open
    loaded.add_examples(["tr0_1"], ["C0"])
    loaded.save(str(tmp_path), include_onnx=False)
    assert load(tmp_path).training_history["C0"] == 13


def test_saved_files_end_with_a_newline(clf, tmp_path):
    clf.save(str(tmp_path), include_onnx=False)
    assert (tmp_path / "config.json").read_bytes().endswith(b"\n")
    assert (tmp_path / "examples.json").read_bytes().endswith(b"\n")


def test_saving_all_examples_restores_the_memory_exactly(clf, tmp_path):
    clf.save(str(tmp_path), include_onnx=False, all_examples=True)
    loaded = load(tmp_path)
    for label, examples in clf.memory.examples.items():
        held = loaded.memory.examples[label]
        assert [ex.text for ex in held] == [ex.text for ex in examples]       # order kept
        assert torch.allclose(torch.stack([ex.embedding for ex in held]),
                              torch.stack([ex.embedding for ex in examples]))
    assert loaded.memory.prototype_base == {}                                 # nothing summarised
    assert loaded.memory.saved_radii == {}
    assert loaded.memory.class_radius("C0") == pytest.approx(clf.memory.class_radius("C0"))
    for (a, sa), (b, sb) in zip(clf.predict("tr0_3"), loaded.predict("tr0_3")):
        assert a == b and sa == pytest.approx(sb, abs=1e-5)


# --- 2. calibration bookkeeping ---------------------------------------------------------------------

def test_calibration_time_and_examples_since_are_tracked_and_saved(clf, data, tmp_path):
    assert clf.last_calibrated_at is None
    assert clf.examples_since_calibration == 24

    texts, labels = data.classes(15, n_classes=2, noise=0.3, seed=7, prefix="cal")
    report = clf.calibrate(texts, labels)
    assert {"ece_before", "ece_after", "temperature", "calibrated_at"} <= set(report)
    stamp = datetime.fromisoformat(clf.last_calibrated_at)
    assert stamp.utcoffset().total_seconds() == 0
    assert report["calibrated_at"] == clf.last_calibrated_at
    assert clf.examples_since_calibration == 0

    clf.add_examples(["tr0_0", "tr1_0", "tr1_1"], ["C0", "C1", "C1"])
    assert clf.examples_since_calibration == 3

    clf.save(str(tmp_path), include_onnx=False)
    loaded = load(tmp_path)
    assert loaded.last_calibrated_at == clf.last_calibrated_at
    assert loaded.examples_since_calibration == 3


def test_a_failed_update_does_not_count_its_examples(clf, monkeypatch):
    before = clf.examples_since_calibration
    monkeypatch.setattr(clf, "_train_adaptive_head", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        clf.add_examples(["tr0_0"], ["C0"])
    assert clf.examples_since_calibration == before


def test_calibration_uses_batched_predictions(clf, data, monkeypatch):
    texts, labels = data.classes(15, n_classes=2, noise=0.3, seed=7, prefix="cal")
    single = []
    monkeypatch.setattr(clf, "_predict_regular", lambda *a, **k: single.append(1) or [])
    clf.calibrate(texts, labels)
    assert single == []


# --- 3. retention: newest plus boundary examples -------------------------------------------------------

def at(x, y=1.0, dim=8):
    v = torch.zeros(dim)
    v[0], v[1] = x, y
    return v


def held(memory_or_clf, label):
    memory = getattr(memory_or_clf, "memory", memory_or_clf)
    return [ex.text for ex in memory.examples[label]]


def boundary_memory(**config):
    """Class B sits at x=5; A's examples sit at x=0 except a few that lean towards B."""
    memory = PrototypeMemory(8, ModelConfig({'max_examples_per_class': 6, **config}))
    memory.add_example(Example("b", "B", at(5.0, 0.0)), "B")
    lean = {1: 1.0, 2: 4.5, 5: 4.0, 7: 3.5}
    for i in range(12):
        memory.add_example(Example(f"a{i}", "A", at(lean.get(i, 0.0))), "A")
    memory._rebuild_index()
    return memory, lean


def test_pruning_keeps_the_newest_and_the_boundary_examples():
    memory, lean = boundary_memory()
    # Half the cap (3) for the newest, the rest for the older examples nearest class B.
    assert held(memory, "A") == ["a2", "a5", "a7", "a9", "a10", "a11"]
    # Keeping the examples nearest the class mean would have dropped all three borderline ones.
    kept = torch.stack([ex.embedding for ex in memory.examples["A"]])
    assert torch.allclose(memory.prototypes["A"], kept.mean(dim=0), atol=1e-6)
    assert memory.get_nearest_prototypes(at(0.0), k=1)[0][0] == "A"
    assert memory.get_nearest_prototypes(at(5.0, 0.0), k=1)[0][0] == "B"


def test_the_recent_share_is_configurable():
    memory, _ = boundary_memory(retention_recent_fraction=1.0)
    assert held(memory, "A") == [f"a{i}" for i in range(6, 12)]
    memory, _ = boundary_memory(retention_recent_fraction=0.0)
    # Boundary only: the four leaning examples, then ties broken by age (oldest first).
    assert held(memory, "A") == ["a0", "a1", "a2", "a3", "a5", "a7"]
    with pytest.raises(ValueError):
        ModelConfig({'retention_recent_fraction': 1.5})


def test_a_single_class_keeps_its_newest_examples():
    memory = PrototypeMemory(8, ModelConfig({'max_examples_per_class': 5}))
    for i in range(12):
        memory.add_example(Example(f"t{i}", "A", at(float(i))), "A")
    assert held(memory, "A") == [f"t{i}" for i in range(7, 12)]


def test_classifier_retention_keeps_prototype_index_and_head_consistent(new_classifier, data):
    clf = new_classifier(head_steps=20, max_examples_per_class=10)
    old = [data.add(f"old{i}", 0, 0.4, i) for i in range(10)]
    other = [data.add(f"b{i}", 1, 0.1, 50 + i) for i in range(10)]
    clf.add_examples(old + other, ["C0"] * 10 + ["C1"] * 10)
    c1 = clf.memory.prototypes["C1"].clone()
    # The "C0" topic drifts to a new axis.
    new = [data.add(f"new{i}", 2, 0.1, 100 + i) for i in range(10)]
    clf.add_examples(new, ["C0"] * 10)

    candidates = old + new[:5]
    by_closeness = sorted(candidates, key=lambda t: float(torch.norm(data.store[t] - c1)))
    assert held(clf, "C0")[5:] == new[5:]                                  # the newest 5
    assert set(held(clf, "C0")[:5]) == set(by_closeness[:5])               # the 5 nearest C1
    expected = torch.stack([data.store[t] for t in held(clf, "C0")]).mean(dim=0)
    assert torch.allclose(clf.memory.prototypes["C0"], expected, atol=1e-6)
    assert clf.memory.index.ntotal == 2
    probe = data.add("probe", 2, 0.1, 999)
    assert clf.predict(probe)[0][0] == "C0"
    assert clf.memory.ood_score(data.store[probe])[1] == "C0"
    assert clf.training_history["C0"] == 20                 # all-time count, not what is held


def test_a_reloaded_class_switches_to_its_held_examples_at_the_first_eviction(new_classifier, data, tmp_path):
    clf = new_classifier(head_steps=20, max_examples_per_class=10)
    texts, labels = data.classes(10, n_classes=2, noise=0.2, seed=1)
    clf.add_examples(texts, labels)
    clf.save(str(tmp_path), include_onnx=False)              # 5 representative examples per class
    loaded = load(tmp_path)
    assert "C0" in loaded.memory.prototype_base and "C0" in loaded.memory.saved_radii

    first = [data.add(f"n{i}", 0, 0.2, 300 + i) for i in range(4)]
    loaded.add_examples(first, ["C0"] * 4)                  # 9 held: no eviction yet
    base, _ = loaded.memory.prototype_base["C0"]
    blended = (base * 6 + torch.stack([data.store[t] for t in first]).sum(dim=0)) / 10
    assert torch.allclose(loaded.memory.prototypes["C0"], blended, atol=1e-6)

    rest = [data.add(f"n{i}", 0, 0.2, 300 + i) for i in range(4, 10)]
    loaded.add_examples(rest, ["C0"] * 6)
    assert len(loaded.memory.examples["C0"]) == 10
    assert held(loaded, "C0")[5:] == rest[1:]
    assert "C0" not in loaded.memory.prototype_base and "C0" not in loaded.memory.post_sum
    assert "C0" not in loaded.memory.post_n and "C0" not in loaded.memory.saved_radii
    expected = torch.stack([ex.embedding for ex in loaded.memory.examples["C0"]]).mean(dim=0)
    assert torch.allclose(loaded.memory.prototypes["C0"], expected, atol=1e-6)


def test_lowering_the_limit_prunes_and_retrains(clf, monkeypatch):
    trained = []
    real = clf._train_adaptive_head
    monkeypatch.setattr(clf, "_train_adaptive_head", lambda *a, **k: trained.append(1) or real(*a, **k))
    c1 = clf.memory.prototypes["C1"].clone()
    older = [ex for ex in clf.memory.examples["C0"] if ex.text not in ("tr0_10", "tr0_11")]
    nearest = sorted(older, key=lambda ex: float(torch.norm(ex.embedding - c1)))[:2]

    assert clf.set_max_examples_per_class(4) == 16
    assert {label: len(ex) for label, ex in clf.memory.examples.items()} == {"C0": 4, "C1": 4}
    assert held(clf, "C0")[2:] == ["tr0_10", "tr0_11"]
    assert set(held(clf, "C0")[:2]) == {ex.text for ex in nearest}
    assert trained == [1]
    assert clf.set_max_examples_per_class(100) == 0
    with pytest.raises(ValueError):
        clf.set_max_examples_per_class(0)


def test_a_failed_update_restores_evicted_examples(new_classifier, data, monkeypatch):
    clf = new_classifier(head_steps=20, max_examples_per_class=12)
    texts, labels = data.classes(12, n_classes=2, noise=0.2, seed=1)
    clf.add_examples(texts, labels)
    held = [ex.text for ex in clf.memory.examples["C0"]]
    monkeypatch.setattr(clf, "_train_adaptive_head", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        clf.add_examples([data.add("x", 0, 0.2, 5)], ["C0"])
    assert [ex.text for ex in clf.memory.examples["C0"]] == held


def test_invalid_limits_are_rejected():
    with pytest.raises(ValueError):
        ModelConfig({'max_examples_per_class': 0})


# --- 4 and 5. batched OOD scores and the percentile radius -------------------------------------------

def test_batched_ood_scores_match_single_scores_from_one_encoder_pass(clf, data, monkeypatch):
    texts = ["tr0_0", "tr1_5", data.add("far", 9, 0.1, 5)]
    expected = [clf.ood_score(t) for t in texts]
    single = clf.predict_batch(texts, k=2)

    calls = []
    real = AdaptiveClassifier._get_embeddings
    monkeypatch.setattr(AdaptiveClassifier, "_get_embeddings",
                        lambda self, batch: calls.append(len(batch)) or real(self, batch))
    predictions, scores = clf.predict_batch_with_ood(texts, k=2)
    assert calls == [3]
    assert scores == pytest.approx(expected)
    assert predictions == single
    assert scores[2] > 1.25 > scores[0]


def test_the_radius_is_a_percentile_of_training_distances(new_classifier, data):
    clf = new_classifier(head_steps=20)
    texts, labels = data.classes(100, n_classes=2, noise=0.3, seed=3)
    clf.add_examples(texts, labels)
    proto = clf.memory.prototypes["C0"]
    distances = torch.norm(torch.stack([data.store[t] for t in texts[:100]]) - proto, dim=1).numpy()
    assert clf.memory.class_radius("C0") == pytest.approx(np.percentile(distances, 95), rel=1e-5)
    assert clf.memory.class_radius("C0") < distances.max()

    # About 5% of a class's own training examples score above 1.0.
    own = clf.memory.ood_scores(torch.stack([data.store[t] for t in texts[:100]]))[0]
    assert 1 <= sum(score > 1.0 for score in own) <= 6

    clf.config.ood_radius_percentile = 100
    assert clf.memory.class_radius("C0") == pytest.approx(distances.max(), rel=1e-5)


def test_one_outlier_does_not_widen_the_radius(new_classifier, data):
    clf = new_classifier(head_steps=20)
    texts, labels = data.classes(40, n_classes=2, noise=0.1, seed=3)
    clf.add_examples(texts, labels)
    before = clf.memory.class_radius("C0")
    clf.add_examples([data.add("outlier", 7, 0.0, 0)], ["C0"])
    assert clf.memory.class_radius("C0") < 1.5 * before


def test_radius_percentile_round_trips_and_is_validated(clf, tmp_path):
    clf.config.ood_radius_percentile = 90.0
    clf.save(str(tmp_path), include_onnx=False)
    assert load(tmp_path).config.ood_radius_percentile == 90.0
    with pytest.raises(ValueError):
        ModelConfig({'ood_radius_percentile': 0})
    with pytest.raises(ValueError):
        ModelConfig({'ood_radius_percentile': 101})
