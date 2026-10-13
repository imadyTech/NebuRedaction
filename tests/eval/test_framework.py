"""Tests for the framework's contracts: determinism, slices, transforms, systems."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from datagen.unit import DetectionUnit
from eval import testsets, transforms
from eval.baselines import available_systems, build
from eval.baselines.rules import RuleBaseline, classify_key, split_key_words

# Imported under an alias: pytest tries to collect any imported name starting
# with Test, and TestSet is a dataclass, not a test class.
from eval.evaluate import TestSet as EvalSet
from eval.evaluate import run_system, truth_of
from eval.report import build_report, render_markdown
from eval.slices import slices_for
from eval.system import SystemOutput


def unit(
    unit_id: str,
    key: str,
    value: str,
    label: str | None = "none",
    **kwargs,
) -> DetectionUnit:
    """Build a unit for a test."""
    return DetectionUnit(id=unit_id, key=key, value=value, source="synth", label=label, **kwargs)


@pytest.fixture
def tiny_set() -> EvalSet:
    """A hand-built test set with one of each interesting shape."""
    return EvalSet(
        name="tiny",
        units=[
            unit("1", "ApiKey", "AKIA" + "Q" * 16, "api_key", family="aws.access_key_id"),
            unit("2", "TraceId", "a" * 32, "none", family="neg.trace_id"),
            unit(
                "3",
                "tokenCount",
                "128",
                "none",
                family="neg.count_integer",
                meta={"adversarial": True},
            ),
            unit(
                "4",
                "Value",
                "b" * 40,
                "api_key",
                family="generic.high_entropy_key",
                meta={"held_out_family": True},
            ),
            unit("5", "IdToken", "c" * 32, "token", family="ambiguous.hex32", ambiguous=True),
        ],
        description="hand-built",
    )


class TestSystemOutput:
    def test_redaction_score_is_the_max_over_secret_classes(self):
        output = SystemOutput("s", "continuous", [{"api_key": 0.4, "token": 0.7, "none": 0.9}])
        # 'none' must not contribute: it is not a secret class.
        assert output.redaction_scores() == [0.7]

    def test_empty_scores_mean_zero(self):
        assert SystemOutput("s", "continuous", [{}]).redaction_scores() == [0.0]

    def test_unknown_class_is_rejected(self):
        with pytest.raises(ValueError, match="unknown classes"):
            SystemOutput("s", "continuous", [{"pii": 1.0}])

    def test_binary_system_must_emit_zero_or_one(self):
        with pytest.raises(ValueError, match=r"must emit 0\.0 or 1\.0"):
            SystemOutput("s", "binary", [{"api_key": 0.5}])

    def test_predicted_class_is_none_when_nothing_scored(self):
        assert SystemOutput("s", "binary", [{}]).predicted_class(0) is None

    def test_predicted_class_breaks_ties_deterministically(self):
        output = SystemOutput("s", "continuous", [{"api_key": 0.5, "token": 0.5}])
        assert output.predicted_class(0) == output.predicted_class(0)


class TestTruth:
    def test_unlabelled_units_are_refused(self):
        with pytest.raises(ValueError, match="unlabelled"):
            truth_of([unit("1", "k", "v", None)])

    def test_secret_labels_are_positive(self):
        assert truth_of([unit("1", "k", "v", "api_key"), unit("2", "k", "v", "none")]) == [
            True,
            False,
        ]


class TestTransforms:
    def test_every_transform_is_deterministic(self):
        # Assembled rather than written as a literal: a string matching a real
        # vendor format is a secret-scanner finding wherever it appears,
        # including in test source (NR-0001 §9.16).
        value = "AKIA" + "TESTVALUE1234567"[:16]
        for transform in transforms.TRANSFORMS:
            assert transform.apply(value) == transform.apply(value)

    def test_base64_wrap_is_decodable(self):
        import base64

        wrapped = transforms.base64_wrap("secret-value")
        assert base64.b64decode(wrapped).decode() == "secret-value"

    def test_case_flip_destroys_a_case_sensitive_prefix(self):
        assert transforms.case_flip("AKIA" + "Q" * 16).startswith("akia")

    def test_truncate_keeps_at_least_eight_characters(self):
        assert len(transforms.truncate("abcdefghij")) >= 8
        assert transforms.truncate("short") == "short"

    def test_affix_noise_adds_to_both_ends(self):
        value = "AKIA" + "Q" * 16
        decorated = transforms.affix_noise(value)
        assert value in decorated and decorated != value

    def test_applying_a_transform_preserves_labels_and_marks_the_slice(self, tiny_set):
        transformed = transforms.apply_transform(
            tiny_set.units,
            transforms.TRANSFORMS[1],  # base64_wrapped
        )
        assert [u.label for u in transformed] == [u.label for u in tiny_set.units]
        assert all(u.meta["transform"] == "base64_wrapped" for u in transformed)
        assert all("#base64_wrapped" in u.id for u in transformed)

    def test_canonical_is_the_identity(self, tiny_set):
        same = transforms.apply_transform(tiny_set.units, transforms.TRANSFORMS[0])
        assert [u.value for u in same] == [u.value for u in tiny_set.units]


class TestSlices:
    def test_empty_slices_are_dropped(self, tiny_set):
        names = {s.name for s in slices_for(tiny_set.units)}
        assert "all" in names
        assert "held_out_families" in names
        assert "adversarial_negatives" in names
        # No multi-line values in this set, so that slice must be absent rather
        # than present-and-empty.
        assert "multiline_values" not in names

    def test_key_recovery_slices_appear_only_when_the_metadata_does(self, tiny_set):
        assert not any(s.name.startswith("key_recovery_") for s in slices_for(tiny_set.units))
        with_meta = [
            replace(u, meta={**u.meta, "key_recovery": "assignment"}) for u in tiny_set.units
        ]
        assert any(s.name == "key_recovery_assignment" for s in slices_for(with_meta))

    def test_source_slices_appear_only_for_mixed_sources(self, tiny_set):
        assert not any(s.name.startswith("source_") for s in slices_for(tiny_set.units))
        mixed = [*tiny_set.units, replace(tiny_set.units[0], id="x", source="third_party:creddata")]
        assert any(s.name == "source_third_party_creddata" for s in slices_for(mixed))


class TestRuleBaseline:
    def test_key_words_split_across_casing_conventions(self):
        assert split_key_words("AWS_SECRET_ACCESS_KEY") == ["aws", "secret", "access", "key"]
        assert split_key_words("apiKey") == ["api", "key"]
        assert split_key_words("X-Api-Key") == ["x", "api", "key"]
        assert split_key_words("<slot:user>") == ["user"]

    @pytest.mark.parametrize(
        "key",
        ["Password", "DB_PASSWORD", "apiKey", "ClientSecret", "Authorization", "PrivateKey"],
    )
    def test_sensitive_key_names_are_recognised(self, key):
        assert classify_key(key) is not None

    @pytest.mark.parametrize(
        "key",
        [
            "tokenCount",
            "PasswordPolicy",
            "KeyCount",
            "PartitionKey",
            "ContentHash",
            "PublicKey",
            "TokenExpiresIn",
            "certificateThumbprint",
            "PasswordLastChanged",
            "secretary",
        ],
    )
    def test_misleading_key_names_do_not_fire(self, key):
        """This is the adversarial slice; a baseline that fires here is the bug."""
        assert classify_key(key) is None

    def test_substring_matching_is_not_used(self):
        # 'keyboard' contains 'key' but is not a key name.
        assert classify_key("keyboardLayout") is None

    def test_value_patterns_beat_key_names(self):
        baseline = RuleBaseline()
        # A real AWS key under a misleading name is still a secret.
        label, rule = baseline.decide(unit("1", "tokenCount", "AKIA" + "Q" * 16, "api_key"))
        assert label == "api_key"
        assert rule == "aws.access_key_id"

    def test_output_is_binary_and_well_formed(self, tiny_set):
        output = RuleBaseline().score(tiny_set.units)
        assert output.kind == "binary"
        assert len(output.scores) == len(tiny_set.units)

    def test_values_only_variant_ignores_key_names(self):
        values_only = build("rules-values-only")
        label, _ = values_only.decide(unit("1", "Password", "hunter2", "password"))
        assert label is None

    def test_every_rule_records_its_provenance(self):
        from eval.baselines.rules import VALUE_RULES

        for rule in VALUE_RULES:
            assert rule.provenance
            assert (
                rule.provenance.startswith("http")
                or "gitleaks" in rule.provenance
                or ("Derived" in rule.provenance)
            )

    def test_ported_rules_name_their_source(self):
        """NR-0000's licence review requires attribution for ported patterns."""
        from eval.baselines.rules import VALUE_RULES

        ported = [r for r in VALUE_RULES if "gitleaks" in r.provenance]
        assert ported, "the baseline is specified to port a gitleaks subset"
        for rule in ported:
            assert "MIT" in rule.provenance


class TestRegistry:
    def test_every_control_group_is_registered(self):
        names = available_systems()
        for expected in ("rules", "rules+entropy", "credsweeper", "laya", "student-a"):
            assert expected in names

    def test_sdlog_is_absent_on_purpose(self):
        """Plan §5.1 item 0: not used in v0.1, so there is nothing to invoke."""
        assert not any("sdlog" in name.lower() for name in available_systems())

    def test_unknown_system_lists_what_exists(self):
        with pytest.raises(KeyError, match="registered:"):
            build("no-such-system")

    def test_placeholders_score_nothing_and_say_so(self, tiny_set):
        """The student models are still placeholders; NR-0004 fills them."""
        output = build("student-a").score(tiny_set.units)
        assert all(row == {} for row in output.scores)
        assert "placeholder" in output.notes["status"]
        assert "not a measurement" in output.notes["warning"]

    def test_control_group_4_is_no_longer_a_placeholder(self):
        """NR-0003 implemented it, as a replay of precomputed probabilities.

        It cannot be built from a bare name, because the adapter needs the path
        to the probabilities for the corpus being evaluated and guessing one
        would silently evaluate the wrong file. The registry entry exists so
        that `--systems laya` explains itself instead of raising a KeyError.
        """
        with pytest.raises(ValueError, match="control_group_4"):
            build("laya")


class TestDeterminism:
    """NR-0002 §2.1 item 4 and §6: two runs on the same input give the same report."""

    def _report(self, tiny_set: EvalSet) -> dict:
        selection = EvalSet(name="sel", units=tiny_set.units, description="selection")
        run = run_system(RuleBaseline(), selection, [tiny_set])
        return build_report(
            [run], [tiny_set], selection, schema_version="1.0", testsets_lock={"testsets": []}
        )

    def test_report_json_is_identical_across_runs(self, tiny_set):
        first = json.dumps(self._report(tiny_set), sort_keys=False)
        second = json.dumps(self._report(tiny_set), sort_keys=False)
        assert first == second

    def test_report_contains_no_timestamp(self, tiny_set):
        """A timestamp in report.json would break the reproducibility requirement."""
        text = json.dumps(self._report(tiny_set))
        for marker in ("generated_utc", "duration_seconds", "platform"):
            assert marker not in text

    def test_run_meta_carries_what_report_json_cannot(self):
        from eval.report import build_run_meta

        meta = build_run_meta(1.25, ["rules"])
        assert "generated_utc" in meta and "duration_seconds" in meta

    def test_continuous_system_is_also_deterministic(self, tiny_set):
        selection = EvalSet(name="sel", units=tiny_set.units, description="selection")
        first = run_system(build("rules+entropy"), selection, [tiny_set])
        second = run_system(build("rules+entropy"), selection, [tiny_set])
        assert first.thresholds == second.thresholds
        assert json.dumps(first.results) == json.dumps(second.results)

    def test_markdown_renders_without_error(self, tiny_set):
        text = render_markdown(self._report(tiny_set))
        assert "# Evaluation report" in text
        assert "Threshold selection" in text


class TestTestsetLock:
    def test_lock_pins_a_digest_per_set(self):
        lock = testsets.read_lock()
        assert lock["schema_version"] == "1.0"
        names = {entry["name"] for entry in lock["testsets"]}
        assert {"synth_val", "synth_test", "creddata"} <= names

    def test_exactly_one_selection_set_is_declared(self):
        roles = [entry["role"] for entry in testsets.read_lock()["testsets"]]
        assert roles.count("selection") == 1

    def test_the_selection_set_is_synthetic(self):
        """EXECUTION.md §8: thresholds may never be chosen on third-party data."""
        selection = [
            entry for entry in testsets.read_lock()["testsets"] if entry["role"] == "selection"
        ]
        assert selection[0]["path"].startswith("data/synth/")

    def test_own_logs_are_not_a_test_set_yet(self):
        """NR-0002 §2.3A item 4: unlabelled until NR-0003."""
        paths = [entry["path"] for entry in testsets.read_lock()["testsets"]]
        assert not any("data/own" in path for path in paths)

    def test_verify_reports_drift_rather_than_raising(self, tmp_path):
        broken = tmp_path / "testsets.lock"
        broken.write_text(
            json.dumps(
                {
                    "lock_version": 1,
                    "schema_version": "1.0",
                    "testsets": [
                        {
                            "name": "synth_test",
                            "path": "data/synth/v0/test.jsonl",
                            "sha256": "0" * 64,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        problems = testsets.verify_lock(broken)
        assert any("digest changed" in problem for problem in problems)

    def test_labelled_only_drops_unlabelled_units(self):
        kept = testsets.labelled_only([unit("1", "k", "v", "api_key"), unit("2", "k", "v", None)])
        assert [u.id for u in kept] == ["1"]
