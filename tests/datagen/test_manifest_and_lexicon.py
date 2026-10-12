"""Tests for the synthetic-sample manifest and the key-name lexicon."""

from __future__ import annotations

import json

import pytest
from datagen.lexicon import (
    CONTEXT_HINTING,
    CONTEXT_MISLEADING,
    CONTEXT_NEUTRAL,
    LEXICON_PATH,
    load_lexicon,
)
from datagen.manifest import (
    MANIFEST_VERSION,
    collect_digests,
    load_manifest,
    value_digest,
    write_manifest,
)
from datagen.unit import DetectionUnit


def _unit(label: str, value: str, unit_id: str = "a") -> DetectionUnit:
    return DetectionUnit(id=unit_id, key="k", value=value, source="synth", label=label)


class TestManifest:
    def test_digest_is_sha256_of_the_utf8_bytes(self):
        import hashlib

        assert value_digest("abc") == hashlib.sha256(b"abc").hexdigest()

    def test_values_are_hashed_verbatim(self):
        """No trimming or case folding: a near miss must not be accepted."""
        assert value_digest("abc") != value_digest(" abc")
        assert value_digest("abc") != value_digest("ABC")

    def test_only_secret_values_are_registered(self):
        digests = collect_digests([_unit("api_key", "secret-1", "a"), _unit("none", "plain", "b")])
        assert value_digest("secret-1") in digests
        assert value_digest("plain") not in digests

    def test_unlabelled_units_are_not_registered(self):
        unit = DetectionUnit(id="a", key="k", value="mystery", source="synth")
        assert collect_digests([unit]) == set()

    def test_multiline_values_register_their_lines_too(self):
        value = "-----BEGIN RSA PRIVATE KEY-----\nAAAA\nBBBB\n-----END RSA PRIVATE KEY-----"
        digests = collect_digests([_unit("private_key", value)])
        assert value_digest(value) in digests
        assert value_digest("AAAA") in digests
        assert value_digest("BBBB") in digests

    def test_blank_lines_in_a_multiline_value_are_not_registered(self):
        digests = collect_digests([_unit("private_key", "AAAA\n\nBBBB")])
        assert value_digest("") not in digests

    def test_write_and_load_round_trip(self, tmp_path):
        path = tmp_path / "manifest.json"
        count = write_manifest(
            [_unit("api_key", "s1", "a"), _unit("token", "s2", "b")],
            path,
            seed=7,
            config_name="test",
        )
        assert count == 2
        assert load_manifest(path) == {value_digest("s1"), value_digest("s2")}

    def test_the_manifest_never_stores_values(self, tmp_path):
        path = tmp_path / "manifest.json"
        write_manifest([_unit("api_key", "topsecretvalue")], path, seed=1, config_name="t")
        assert "topsecretvalue" not in path.read_text(encoding="utf-8")

    def test_the_manifest_records_its_provenance(self, tmp_path):
        path = tmp_path / "manifest.json"
        write_manifest([_unit("api_key", "s")], path, seed=42, config_name="v0")
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["seed"] == 42
        assert document["config_name"] == "v0"
        assert document["algorithm"] == "sha256"
        assert document["manifest_version"] == MANIFEST_VERSION

    def test_digests_are_sorted_so_the_file_is_stable(self, tmp_path):
        path = tmp_path / "manifest.json"
        write_manifest(
            [_unit("api_key", f"s{index}", str(index)) for index in range(20)],
            path,
            seed=1,
            config_name="t",
        )
        digests = json.loads(path.read_text(encoding="utf-8"))["digests"]
        assert digests == sorted(digests)

    def test_a_newer_manifest_version_is_refused(self, tmp_path):
        path = tmp_path / "manifest.json"
        path.write_text(
            json.dumps({"manifest_version": MANIFEST_VERSION + 1, "digests": []}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="newer than this checker"):
            load_manifest(path)

    def test_a_non_manifest_file_is_refused(self, tmp_path):
        path = tmp_path / "manifest.json"
        path.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="not a synthetic-sample manifest"):
            load_manifest(path)

    def test_a_manifest_covers_every_secret_in_the_corpus_it_was_built_from(
        self, fixture_corpus, fixture_manifest_digests
    ):
        """A corpus and its manifest are generated together and must stay in step."""
        from datagen.loader import iter_units

        checked = 0
        for split in ("train", "val", "test"):
            for unit in iter_units(fixture_corpus[split]):
                if unit.should_redact:
                    assert value_digest(unit.value) in fixture_manifest_digests, unit.id
                    checked += 1
        assert checked > 0

    def test_a_manifest_does_not_register_non_secrets(
        self, fixture_corpus, fixture_manifest_digests
    ):
        """Registering a negative would let an unrelated string pass the fixture guard."""
        from datagen.loader import iter_units

        for unit in iter_units(fixture_corpus["train"]):
            if unit.label == "none":
                assert value_digest(unit.value) not in fixture_manifest_digests, unit.id


class TestLexicon:
    @pytest.fixture(scope="class")
    @classmethod
    def lexicon(cls):
        return load_lexicon()

    def test_all_three_groups_are_populated(self, lexicon):
        assert lexicon.sensitive
        assert lexicon.neutral
        assert lexicon.misleading
        assert lexicon.scopes

    def test_every_secret_class_has_sensitive_key_names(self, lexicon):
        from datagen.unit import SECRET_LABELS

        assert SECRET_LABELS.issubset(set(lexicon.sensitive))

    def test_sensitive_for_an_unknown_class_falls_back_to_everything(self, lexicon):
        assert set(lexicon.sensitive_for("none")) == set(lexicon.all_sensitive())

    def test_misleading_names_all_contain_a_credential_word(self, lexicon):
        """That is what makes them traps; a name without one belongs in `neutral`."""
        words = (
            "token",
            "secret",
            "password",
            "key",
            "hash",
            "credential",
            "auth",
            "login",
            "thumbprint",
            "certificate",
        )
        for name in lexicon.misleading:
            lowered = name.lower()
            assert any(word in lowered for word in words), name

    def test_neutral_names_contain_no_credential_word(self, lexicon):
        """A neutral name that hints would quietly weaken the neutral-context cases."""
        words = ("token", "secret", "password", "passwd", "pwd", "apikey", "credential")
        for name in lexicon.neutral:
            lowered = name.lower().replace("_", "").replace("-", "")
            assert not any(word in lowered for word in words), name

    def test_sensitive_and_misleading_groups_do_not_overlap(self, lexicon):
        assert not set(lexicon.all_sensitive()) & set(lexicon.misleading)

    def test_neutral_group_overlaps_neither(self, lexicon):
        neutral = set(lexicon.neutral)
        assert not neutral & set(lexicon.all_sensitive())
        assert not neutral & set(lexicon.misleading)

    def test_names_cover_several_casing_conventions(self, lexicon):
        """Plan §4.2 needs the key tokeniser exercised on all four conventions."""
        names = set(lexicon.all_sensitive())
        assert any("_" in name and name.isupper() for name in names), "SCREAMING_SNAKE"
        assert any("_" in name and name.islower() for name in names), "snake_case"
        assert any("-" in name for name in names), "kebab-case"
        assert any(name[0].isupper() and "_" not in name for name in names), "PascalCase"

    @pytest.mark.parametrize("context", [CONTEXT_HINTING, CONTEXT_NEUTRAL, CONTEXT_MISLEADING])
    def test_pick_key_returns_a_name_from_the_right_group(self, lexicon, context):
        import random

        rng = random.Random(1)
        groups = {
            CONTEXT_HINTING: set(lexicon.sensitive_for("api_key")),
            CONTEXT_NEUTRAL: set(lexicon.neutral),
            CONTEXT_MISLEADING: set(lexicon.misleading),
        }
        for _ in range(50):
            assert lexicon.pick_key(rng, "api_key", context) in groups[context]

    def test_pick_key_rejects_an_unknown_context(self, lexicon):
        import random

        with pytest.raises(ValueError, match="unknown context"):
            lexicon.pick_key(random.Random(1), "api_key", "sideways")

    def test_the_lexicon_file_records_its_provenance(self):
        """NR-0001 §6: the vocabulary must not be derived from third-party corpora.

        That is a claim about where the names came from, which no runtime check
        can establish. What this test can do is make sure the file states its
        sources and names the prohibition, so the constraint travels with the
        data instead of living only in a work order.
        """
        text = LEXICON_PATH.read_text(encoding="utf-8")
        assert "PROVENANCE" in text
        assert "LogHub" in text and "CredData" in text and "SDLog" in text
        assert "not used at all in v0.1" in text or "never" in text.lower()

    def test_a_lexicon_with_an_empty_group_is_refused(self, tmp_path):
        import yaml

        document = yaml.safe_load(LEXICON_PATH.read_text(encoding="utf-8"))
        document["misleading"] = []
        path = tmp_path / "keys.yaml"
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
        with pytest.raises(ValueError, match="must not be empty"):
            load_lexicon(path)

    def test_a_lexicon_without_sensitive_names_is_refused(self, tmp_path):
        import yaml

        document = yaml.safe_load(LEXICON_PATH.read_text(encoding="utf-8"))
        document["sensitive"] = {}
        path = tmp_path / "keys.yaml"
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
        with pytest.raises(ValueError, match="non-empty mapping"):
            load_lexicon(path)
