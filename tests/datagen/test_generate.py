"""Tests for the generator: determinism, composition and the split discipline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from datagen.generate import (
    SPLITS,
    build_plan,
    derive_seed,
    generate_split,
    load_config,
    summarise,
    write_split,
)
from datagen.lexicon import CONTEXT_MISLEADING, load_lexicon
from datagen.unit import SECRET_LABELS

CONFIG_PATH = Path("datagen/configs/v0.yaml")
SEED = 20261011
SMALL = 600


@pytest.fixture(scope="module")
def config():
    from dataclasses import replace

    base = load_config(CONFIG_PATH)
    return replace(base, sizes=dict.fromkeys(SPLITS, SMALL))


@pytest.fixture(scope="module")
def lexicon():
    return load_lexicon()


@pytest.fixture(scope="module")
def generated(config, lexicon):
    return {split: generate_split(config, split, SEED, lexicon) for split in SPLITS}


def test_shipped_config_loads():
    assert load_config(CONFIG_PATH).name == "v0"


def test_derive_seed_is_stable_and_distinct():
    assert derive_seed(1, "train", 0) == derive_seed(1, "train", 0)
    assert derive_seed(1, "train", 0) != derive_seed(1, "train", 1)
    assert derive_seed(1, "train", 0) != derive_seed(1, "val", 0)
    assert derive_seed(1, "train", 0) != derive_seed(2, "train", 0)


class TestDeterminism:
    """NR-0001 §2.2 item 7 and §6: the same config and seed must give the same bytes."""

    def test_the_same_seed_gives_byte_identical_output(self, config, lexicon, tmp_path):
        digests = []
        for attempt in range(2):
            units = generate_split(config, "train", SEED, lexicon)
            digests.append(write_split(units, tmp_path / f"attempt{attempt}.jsonl"))
        assert digests[0] == digests[1]
        first = (tmp_path / "attempt0.jsonl").read_bytes()
        second = (tmp_path / "attempt1.jsonl").read_bytes()
        assert first == second

    def test_a_different_seed_gives_different_output(self, config, lexicon, tmp_path):
        a = write_split(generate_split(config, "train", SEED, lexicon), tmp_path / "a.jsonl")
        b = write_split(generate_split(config, "train", SEED + 1, lexicon), tmp_path / "b.jsonl")
        assert a != b

    def test_write_split_digest_matches_the_file(self, config, lexicon, tmp_path):
        path = tmp_path / "out.jsonl"
        digest = write_split(generate_split(config, "val", SEED, lexicon), path)
        assert digest == hashlib.sha256(path.read_bytes()).hexdigest()

    def test_a_single_unit_can_be_regenerated_from_its_recorded_seed(self, generated):
        """Every unit records the seed that made it, so one record is reproducible alone."""
        import random

        from datagen.families import REGISTRY

        for unit in generated["train"][:40]:
            if unit.meta.get("adversarial_carrier") == "template":
                continue
            family = REGISTRY[unit.family]
            replayed = family.generate(random.Random(unit.meta["seed"]))
            assert replayed == unit.value, unit.id


class TestComposition:
    def test_split_sizes_are_exact(self, generated):
        for split, units in generated.items():
            assert len(units) == SMALL, split

    def test_ids_are_unique_within_a_split(self, generated):
        for split, units in generated.items():
            ids = [unit.id for unit in units]
            assert len(set(ids)) == len(ids), split

    def test_every_unit_is_synthetic(self, generated):
        for units in generated.values():
            assert all(unit.source == "synth" for unit in units)

    def test_adversarial_share_of_negatives_meets_the_floor(self, generated):
        """NR-0001 §2.2 item 4: at least 10% of negatives in every split."""
        for split, units in generated.items():
            stats = summarise(units)
            assert stats["adversarial_share_of_negatives"] >= 0.10, (split, stats)

    def test_adversarial_share_matches_the_configured_ratio(self, config, generated):
        """The quota is computed, not sampled, so it should land on the configured value."""
        for split, units in generated.items():
            stats = summarise(units)
            assert stats["adversarial_share_of_negatives"] == pytest.approx(
                config.adversarial_negative_ratio, abs=0.005
            ), split

    def test_adversarial_units_are_all_negative(self, generated):
        """A secret is still a secret under a reassuring name; only negatives are traps."""
        for units in generated.values():
            for unit in units:
                if unit.meta.get("adversarial"):
                    assert unit.label == "none", unit.id

    def test_all_three_adversarial_carriers_are_present(self, generated):
        for split, units in generated.items():
            template_carried = [
                unit for unit in units if unit.meta.get("adversarial_carrier") == "template"
            ]
            key_carried = [
                unit
                for unit in units
                if unit.meta.get("adversarial") and "adversarial_carrier" not in unit.meta
            ]
            assert template_carried, split
            assert key_carried, split

    def test_template_adversarial_units_look_like_the_probe_failure(self, generated):
        """These are the shape NR-0000 §9.5 recorded the teacher failing on."""
        for units in generated.values():
            for unit in units:
                if unit.meta.get("adversarial_carrier") != "template":
                    continue
                assert unit.label == "none"
                assert unit.template is not None
                assert unit.key.startswith("<slot:")
                assert unit.meta.get("context") == CONTEXT_MISLEADING

    def test_every_class_appears_in_every_split(self, generated):
        for split, units in generated.items():
            labels = {unit.label for unit in units}
            assert SECRET_LABELS.issubset(labels), (split, labels)
            assert "none" in labels

    def test_ambiguous_units_are_labelled_and_flagged(self, generated):
        for units in generated.values():
            ambiguous = [unit for unit in units if unit.ambiguous]
            assert ambiguous
            for unit in ambiguous:
                assert unit.family.startswith("ambiguous.")
                # Bias toward redaction (plan §3.3): an ambiguous unit is never
                # labelled 'none'.
                assert unit.label in SECRET_LABELS, unit.id

    def test_all_three_contexts_appear(self, generated):
        for split, units in generated.items():
            contexts = {unit.meta.get("context") for unit in units}
            assert contexts == {"hinting", "neutral", "misleading"}, (split, contexts)

    def test_most_negatives_sit_under_neutral_key_names(self, generated):
        """Real logs are mostly harmless values under harmless names; keep that true here."""
        for split, units in generated.items():
            negatives = [unit for unit in units if unit.label == "none"]
            neutral = [unit for unit in negatives if unit.meta.get("context") == "neutral"]
            assert len(neutral) / len(negatives) > 0.5, split


class TestHeldOutFamilies:
    """Plan §5.3 and NR-0001 §6: held-out families must appear only in the test split."""

    def test_held_out_families_are_absent_from_train_and_val(self, config, generated):
        held_out = config.held_out
        assert held_out, "the shipped config must hold some families out"
        for split in ("train", "val"):
            present = {unit.family for unit in generated[split]}
            leaked = sorted(present & held_out)
            assert not leaked, f"{split} contains held-out families: {leaked}"

    def test_held_out_families_do_appear_in_test(self, config, generated):
        present = {unit.family for unit in generated["test"]}
        missing = sorted(config.held_out - present)
        assert not missing, f"test is missing held-out families: {missing}"

    def test_held_out_units_are_flagged(self, config, generated):
        for unit in generated["test"]:
            expected = unit.family in config.held_out
            assert bool(unit.meta.get("held_out_family")) == expected, unit.id

    def test_train_and_test_share_the_non_held_out_families(self, config, generated):
        train_families = {unit.family for unit in generated["train"]}
        test_families = {unit.family for unit in generated["test"]}
        assert train_families <= test_families
        assert test_families - train_families == config.held_out

    def test_held_out_set_spans_several_classes(self, config):
        """A held-out set confined to one class would not measure much."""
        from datagen.families import REGISTRY

        labels = {REGISTRY[name].label for name in config.held_out_positive}
        assert len(labels) >= 4, labels


class TestConfigValidation:
    def _write(self, tmp_path: Path, text: str) -> Path:
        path = tmp_path / "config.yaml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_unknown_held_out_family_is_rejected(self, tmp_path):
        base = CONFIG_PATH.read_text(encoding="utf-8")
        broken = base.replace("    - neg.ulid", "    - neg.does_not_exist")
        with pytest.raises(ValueError, match="unknown families"):
            load_config(self._write(tmp_path, broken))

    def test_a_positive_listed_as_a_held_out_negative_is_rejected(self, tmp_path):
        base = CONFIG_PATH.read_text(encoding="utf-8")
        broken = base.replace("    - neg.ulid", "    - aws.access_key_id")
        with pytest.raises(ValueError, match="listed as a held-out negative"):
            load_config(self._write(tmp_path, broken))

    def test_adversarial_floor_is_enforced(self, tmp_path):
        base = CONFIG_PATH.read_text(encoding="utf-8")
        broken = base.replace(
            "adversarial_negative_ratio: 0.24", "adversarial_negative_ratio: 0.05"
        )
        with pytest.raises(ValueError, match=r"at least 0\.10"):
            load_config(self._write(tmp_path, broken))

    def test_carrier_split_must_sum_to_one(self, tmp_path):
        base = CONFIG_PATH.read_text(encoding="utf-8")
        broken = base.replace("  template: 0.3", "  template: 0.9")
        with pytest.raises(ValueError, match="sum to"):
            load_config(self._write(tmp_path, broken))

    def test_generator_version_mismatch_is_rejected(self, tmp_path):
        base = CONFIG_PATH.read_text(encoding="utf-8")
        broken = base.replace(
            'generator_version_required: "0.1.0"', 'generator_version_required: "9.9.9"'
        )
        with pytest.raises(ValueError, match="requires generator version"):
            load_config(self._write(tmp_path, broken))

    def test_a_split_too_small_for_every_kind_is_rejected(self, config, lexicon):
        from dataclasses import replace

        tiny = replace(config, sizes=dict.fromkeys(SPLITS, 4))
        import random

        with pytest.raises(ValueError, match="too small"):
            build_plan(tiny, "train", random.Random(0))


class TestSummary:
    def test_summary_counts_add_up(self, generated):
        stats = summarise(generated["train"])
        assert stats["count"] == SMALL
        assert sum(stats["by_label"].values()) == SMALL
        assert sum(stats["by_family"].values()) == SMALL
        assert stats["negatives"] == stats["by_label"]["none"]

    def test_summary_is_json_serialisable(self, generated):
        json.dumps(summarise(generated["test"]))
