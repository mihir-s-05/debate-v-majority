import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from debate_v_majority.tools import rejudge_targets as rt


class FakeEngine:
    def __init__(self, outputs_by_call):
        self.outputs_by_call = list(outputs_by_call)
        self.calls = []

    def generate_batch(self, contexts, batch_size=None, sampling_kwargs=None):
        self.calls.append((contexts, batch_size, sampling_kwargs))
        if not self.outputs_by_call:
            raise AssertionError("Unexpected generate_batch call")
        out = self.outputs_by_call.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


class FakeEngineManager:
    def __init__(self, engine):
        self.engine = engine

    def get_engine(self, model_name):
        return self.engine

    def judge_sampling_kwargs(self, model_name):
        return {"max_tokens": 64}

    def judge_max_new_tokens(self, model_name, sampling_kwargs):
        return 64

    def context_len_tokens(self, model_name):
        return None

    def get_token_counter(self, model_name):
        return None

    def close(self):
        pass


class MinimalManager:
    def __init__(self, *args, **kwargs):
        pass

    def close(self):
        pass


class RejudgeTargetsRefactorTests(unittest.TestCase):
    def _base_row(self, orig_id=1):
        return {
            "orig_id": orig_id,
            "question": "Q?",
            "answer": "A",
            "raw_task": {"question": "Q?", "options": ["A", "B", "C", "D"]},
            "agent_responses": [[{"role": "assistant", "content": "resp"}]],
            "n_rounds": 1,
            "judge_trace": {},
            "final_judge_answer": None,
            "final_judge_correct": 0,
            "final_answer": None,
            "final_correct": 0,
        }

    def test_engine_manager_judge_max_new_tokens_handles_none_override(self):
        mgr = rt.EngineManager(
            gpus="0",
            gpu_memory_utilization=0.9,
            context_len=None,
            enable_yarn=False,
            enforce_eager=False,
            judge_overrides=None,
        )
        mgr._sampling_cfg_by_model["m1"] = rt.SamplingConfig(max_tokens=321)
        mgr._sampling_cfg_by_model["m2"] = rt.SamplingConfig(max_tokens=None)

        self.assertEqual(mgr.judge_max_new_tokens("m1", {"max_tokens": None}), 321)
        self.assertEqual(mgr.judge_max_new_tokens("m1", {"max_tokens": 55}), 55)
        self.assertEqual(mgr.judge_max_new_tokens("m2", None), 4096)

    def test_build_retry_sampling_kwargs_preserves_overrides(self):
        out = rt._build_retry_sampling_kwargs(
            sampling_kwargs={"temperature": 0.3, "top_p": 0.8, "foo": "bar"},
            judge_max_new_tokens=77,
            judge_overrides={"temperature": 0.3, "top_p": 0.8},
        )
        self.assertEqual(out["temperature"], 0.3)
        self.assertEqual(out["top_p"], 0.8)
        self.assertEqual(out["foo"], "bar")
        self.assertEqual(out["max_tokens"], 77)

        out2 = rt._build_retry_sampling_kwargs(
            sampling_kwargs={"temperature": None, "top_p": None},
            judge_max_new_tokens=88,
            judge_overrides={"temperature": None, "top_p": None},
        )
        self.assertEqual(out2["temperature"], 0.0)
        self.assertEqual(out2["top_p"], 1.0)
        self.assertEqual(out2["max_tokens"], 88)

    def test_build_retry_sampling_kwargs_partial_override_stays_deterministic(self):
        out = rt._build_retry_sampling_kwargs(
            sampling_kwargs={"temperature": 0.7, "top_p": 0.9},
            judge_max_new_tokens=64,
            judge_overrides={"max_tokens": 64, "temperature": None, "top_p": None},
        )
        self.assertEqual(out["temperature"], 0.0)
        self.assertEqual(out["top_p"], 1.0)
        self.assertEqual(out["max_tokens"], 64)

    def test_try_reparse_existing_uses_strict_only(self):
        row = self._base_row(orig_id=3)
        row["judge_trace"] = {"judge_raw_response": "raw", "judge_retry_raw_response": "retry"}
        calls = []

        def parse_side_effect(*, dataset, text, raw_task, source_prefix, strict_enabled, recovery_enabled):
            calls.append((source_prefix, strict_enabled, recovery_enabled))
            self.assertTrue(strict_enabled)
            self.assertFalse(recovery_enabled)
            if source_prefix == "raw":
                return rt.JudgeParseResult(answer=None, mode="none", source="none", strict_success=False)
            return rt.JudgeParseResult(answer="B", mode="strict", source="retry_strict", strict_success=True)

        with patch("debate_v_majority.tools.rejudge_targets._parse_judge_text", side_effect=parse_side_effect):
            out = rt._try_reparse_existing(row, dataset="gpqa")

        self.assertEqual(out, ("B", "strict", "retry_strict", False, True))
        self.assertEqual(calls, [("raw", True, False), ("retry", True, False)])

    def test_run_raw_pass_recover_result_goes_to_retry(self):
        row = self._base_row(orig_id=10)
        req = rt.RerunRequest(
            path=Path("/tmp/t.jsonl"),
            orig_id=10,
            row_idx=0,
            row=row,
            before=None,
            model_name="m",
        )
        meta = rt.PreparedRerunMeta(req=req, raw_task=row["raw_task"], judge_context=[{"role": "system", "content": "ctx"}])
        attempts = {}

        def parse_side_effect(*, dataset, text, raw_task, source_prefix, strict_enabled, recovery_enabled):
            self.assertEqual(source_prefix, "raw")
            if recovery_enabled:
                return rt.JudgeParseResult(answer="C", mode="recover", source="raw_recovery", strict_success=False)
            return rt.JudgeParseResult(answer=None, mode="none", source="none", strict_success=False)

        with patch("debate_v_majority.tools.rejudge_targets._parse_judge_text", side_effect=parse_side_effect):
            retry_pending, retry_raw_had_strict = rt._run_raw_pass_for_model(
                dataset="gpqa",
                metas=[meta],
                raw_outputs=["raw_recoverish"],
                attempts=attempts,
            )

        self.assertEqual(retry_pending, [0])
        self.assertEqual(retry_raw_had_strict, [False])
        self.assertEqual(attempts, {})

    def test_run_raw_pass_generation_error_records_attempt(self):
        row = self._base_row(orig_id=11)
        req = rt.RerunRequest(
            path=Path("/tmp/t.jsonl"),
            orig_id=11,
            row_idx=0,
            row=row,
            before=None,
            model_name="m",
        )
        meta = rt.PreparedRerunMeta(req=req, raw_task=row["raw_task"], judge_context=[{"role": "system", "content": "ctx"}])
        attempts = {}

        retry_pending, retry_raw_had_strict = rt._run_raw_pass_for_model(
            dataset="gpqa",
            metas=[meta],
            raw_outputs=[f"[{rt.RAW_ERROR_PREFIX}] boom"],
            attempts=attempts,
        )

        self.assertEqual(retry_pending, [])
        self.assertEqual(retry_raw_had_strict, [False])
        a = attempts[(Path("/tmp/t.jsonl"), 11)]
        self.assertTrue(a.parse_failed)
        self.assertEqual(a.finish_state, "raw_error")
        self.assertEqual(a.retry_reason, "raw_generation_error")

    def test_finalize_retry_pass_recover_result_goes_to_extract(self):
        row = self._base_row(orig_id=12)
        req = rt.RerunRequest(
            path=Path("/tmp/t.jsonl"),
            orig_id=12,
            row_idx=0,
            row=row,
            before=None,
            model_name="m",
        )
        meta = rt.PreparedRerunMeta(req=req, raw_task=row["raw_task"], judge_context=[{"role": "system", "content": "ctx"}])
        attempts = {}

        def parse_side_effect(*, dataset, text, raw_task, source_prefix, strict_enabled, recovery_enabled):
            self.assertEqual(source_prefix, "retry")
            if recovery_enabled:
                return rt.JudgeParseResult(answer="D", mode="recover", source="retry_recovery", strict_success=False)
            return rt.JudgeParseResult(answer=None, mode="none", source="none", strict_success=False)

        with patch("debate_v_majority.tools.rejudge_targets._parse_judge_text", side_effect=parse_side_effect):
            extract_pending = rt._finalize_retry_pass_for_model(
                dataset="gpqa",
                metas=[meta],
                raw_outputs=["raw_bad"],
                retry_pending=[0],
                retry_output_by_meta={0: "retry_recoverish"},
                retry_raw_had_strict=[False],
                attempts=attempts,
            )

        self.assertEqual(extract_pending, [0])
        self.assertEqual(attempts, {})

    def test_finalize_retry_pass_generation_error_records_attempt(self):
        row = self._base_row(orig_id=17)
        req = rt.RerunRequest(
            path=Path("/tmp/t.jsonl"),
            orig_id=17,
            row_idx=0,
            row=row,
            before=None,
            model_name="m",
        )
        meta = rt.PreparedRerunMeta(req=req, raw_task=row["raw_task"], judge_context=[{"role": "system", "content": "ctx"}])
        attempts = {}

        extract_pending = rt._finalize_retry_pass_for_model(
            dataset="gpqa",
            metas=[meta],
            raw_outputs=["raw_bad"],
            retry_pending=[0],
            retry_output_by_meta={0: f"[{rt.RETRY_ERROR_PREFIX}] fail"},
            retry_raw_had_strict=[False],
            attempts=attempts,
        )

        self.assertEqual(extract_pending, [])
        a = attempts[(Path("/tmp/t.jsonl"), 17)]
        self.assertTrue(a.parse_failed)
        self.assertEqual(a.finish_state, "retry_error")
        self.assertEqual(a.retry_reason, "retry_generation_error")

    def test_finalize_retry_pass_strict_success_records_attempt(self):
        row = self._base_row(orig_id=18)
        req = rt.RerunRequest(
            path=Path("/tmp/t.jsonl"),
            orig_id=18,
            row_idx=0,
            row=row,
            before=None,
            model_name="m",
        )
        meta = rt.PreparedRerunMeta(req=req, raw_task=row["raw_task"], judge_context=[{"role": "system", "content": "ctx"}])
        attempts = {}

        def parse_side_effect(*, dataset, text, raw_task, source_prefix, strict_enabled, recovery_enabled):
            self.assertEqual(source_prefix, "retry")
            if recovery_enabled:
                return rt.JudgeParseResult(answer="A", mode="strict", source="retry_strict", strict_success=True)
            return rt.JudgeParseResult(answer="A", mode="strict", source="retry_strict", strict_success=True)

        with patch("debate_v_majority.tools.rejudge_targets._parse_judge_text", side_effect=parse_side_effect):
            extract_pending = rt._finalize_retry_pass_for_model(
                dataset="gpqa",
                metas=[meta],
                raw_outputs=["raw_bad"],
                retry_pending=[0],
                retry_output_by_meta={0: "retry_ok"},
                retry_raw_had_strict=[False],
                attempts=attempts,
            )

        self.assertEqual(extract_pending, [])
        a = attempts[(Path("/tmp/t.jsonl"), 18)]
        self.assertFalse(a.parse_failed)
        self.assertEqual(a.finish_state, "retry_parsed")
        self.assertEqual(a.judged_answer, "A")
        self.assertEqual(a.parse_mode, "strict")
        self.assertFalse(a.used_fallback)
        self.assertTrue(a.retry_had_strict_final)

    def test_finalize_extract_pass_retry_only_strict_success(self):
        row = self._base_row(orig_id=13)
        req = rt.RerunRequest(
            path=Path("/tmp/t.jsonl"),
            orig_id=13,
            row_idx=0,
            row=row,
            before=None,
            model_name="m",
        )
        meta = rt.PreparedRerunMeta(req=req, raw_task=row["raw_task"], judge_context=[{"role": "system", "content": "ctx"}])
        attempts = {}
        engine = FakeEngine(outputs_by_call=[["extract_retry_only_good"]])

        def parse_side_effect(*, dataset, text, raw_task, source_prefix, strict_enabled, recovery_enabled):
            self.assertTrue(strict_enabled)
            self.assertFalse(recovery_enabled)
            if source_prefix == "extract_retry_only":
                return rt.JudgeParseResult(answer="A", mode="strict", source="extract_retry_only_strict", strict_success=True)
            return rt.JudgeParseResult(answer=None, mode="none", source="none", strict_success=False)

        with patch("debate_v_majority.tools.rejudge_targets._parse_judge_text", side_effect=parse_side_effect):
            rt._finalize_extract_pass_for_model(
                dataset="gpqa",
                engine=engine,
                metas=[meta],
                raw_outputs=["raw_bad"],
                extract_pending=[0],
                retry_output_by_meta={0: "retry_bad"},
                retry_raw_had_strict=[False],
                batch_size=8,
                attempts=attempts,
            )

        a = attempts[(Path("/tmp/t.jsonl"), 13)]
        self.assertEqual(a.finish_state, "extract_parsed")
        self.assertEqual(a.retry_reason, "parse_none_then_extract_retry_only")
        self.assertEqual(a.parse_mode, "strict")
        self.assertFalse(a.used_fallback)
        self.assertEqual(len(engine.calls), 1)

    def test_finalize_extract_pass_falls_back_to_stage_b_when_stage_a_unparsed(self):
        row = self._base_row(orig_id=14)
        req = rt.RerunRequest(
            path=Path("/tmp/t.jsonl"),
            orig_id=14,
            row_idx=0,
            row=row,
            before=None,
            model_name="m",
        )
        meta = rt.PreparedRerunMeta(req=req, raw_task=row["raw_task"], judge_context=[{"role": "system", "content": "ctx"}])
        attempts = {}
        engine = FakeEngine(outputs_by_call=[["stage_a_unparsed"], ["stage_b_strict"]])

        def parse_side_effect(*, dataset, text, raw_task, source_prefix, strict_enabled, recovery_enabled):
            self.assertFalse(recovery_enabled)
            if source_prefix == "extract_retry_only":
                return rt.JudgeParseResult(answer=None, mode="none", source="none", strict_success=False)
            if source_prefix == "extract":
                return rt.JudgeParseResult(answer="C", mode="strict", source="extract_strict", strict_success=True)
            return rt.JudgeParseResult(answer=None, mode="none", source="none", strict_success=False)

        with patch("debate_v_majority.tools.rejudge_targets._parse_judge_text", side_effect=parse_side_effect):
            rt._finalize_extract_pass_for_model(
                dataset="gpqa",
                engine=engine,
                metas=[meta],
                raw_outputs=["raw_bad"],
                extract_pending=[0],
                retry_output_by_meta={0: "retry_bad"},
                retry_raw_had_strict=[False],
                batch_size=8,
                attempts=attempts,
            )

        a = attempts[(Path("/tmp/t.jsonl"), 14)]
        self.assertEqual(a.finish_state, "extract_parsed")
        self.assertEqual(a.retry_reason, "parse_none_then_extract")
        self.assertEqual(a.judged_answer, "C")
        self.assertEqual(len(engine.calls), 2)

    def test_finalize_extract_pass_error_path_marks_retry_unparsed(self):
        row = self._base_row(orig_id=15)
        req = rt.RerunRequest(
            path=Path("/tmp/t.jsonl"),
            orig_id=15,
            row_idx=0,
            row=row,
            before=None,
            model_name="m",
        )
        meta = rt.PreparedRerunMeta(req=req, raw_task=row["raw_task"], judge_context=[{"role": "system", "content": "ctx"}])
        attempts = {}
        engine = FakeEngine(
            outputs_by_call=[
                [f"[{rt.EXTRACT_ERROR_PREFIX}] stage_a_error"],
                [f"[{rt.EXTRACT_ERROR_PREFIX}] stage_b_error"],
            ]
        )

        rt._finalize_extract_pass_for_model(
            dataset="gpqa",
            engine=engine,
            metas=[meta],
            raw_outputs=["raw_bad"],
            extract_pending=[0],
            retry_output_by_meta={0: "retry_bad"},
            retry_raw_had_strict=[False],
            batch_size=8,
            attempts=attempts,
        )

        a = attempts[(Path("/tmp/t.jsonl"), 15)]
        self.assertTrue(a.parse_failed)
        self.assertEqual(a.finish_state, "retry_unparsed")
        self.assertEqual(a.parse_mode, "none")
        self.assertEqual(a.parse_source, "none")

    def test_finalize_extract_pass_no_pending_does_nothing(self):
        attempts = {}
        engine = FakeEngine(outputs_by_call=[])
        rt._finalize_extract_pass_for_model(
            dataset="gpqa",
            engine=engine,
            metas=[],
            raw_outputs=[],
            extract_pending=[],
            retry_output_by_meta={},
            retry_raw_had_strict=[],
            batch_size=8,
            attempts=attempts,
        )
        self.assertEqual(attempts, {})
        self.assertEqual(engine.calls, [])

    def test_extract_only_existing_uses_strict_parse_only(self):
        row = self._base_row(orig_id=16)
        row["judge_trace"] = {
            "judge_model": "m1",
            "judge_raw_response": "raw",
            "judge_retry_raw_response": "retry",
        }
        engine = FakeEngine(outputs_by_call=[["extract_output"]])
        manager = FakeEngineManager(engine)

        def parse_side_effect(*, dataset, text, raw_task, source_prefix, strict_enabled, recovery_enabled):
            self.assertEqual(source_prefix, "extract")
            self.assertTrue(strict_enabled)
            self.assertFalse(recovery_enabled)
            return rt.JudgeParseResult(answer=None, mode="none", source="none", strict_success=False)

        with patch("debate_v_majority.tools.rejudge_targets._parse_judge_text", side_effect=parse_side_effect):
            res = rt._extract_only_existing_result(
                dataset="gpqa",
                path=Path("/tmp/t.jsonl"),
                row=row,
                orig_id=16,
                engine_mgr=manager,
            )

        self.assertEqual(res.action, "extract_only_failed")
        self.assertIn("unparsable", res.details.lower())

    def test_plan_target_row_extract_only_missing_model_uses_graceful_path(self):
        row = self._base_row(orig_id=11)
        expected = rt.RowUpdateResult(
            path=Path("/tmp/t.jsonl"),
            orig_id=11,
            action="extract_only_failed",
            before=None,
            after=None,
            changed=False,
            details="Missing judge_trace.judge_model; cannot run extraction-only pass.",
        )
        with (
            patch("debate_v_majority.tools.rejudge_targets._needs_retry", return_value=True),
            patch("debate_v_majority.tools.rejudge_targets._try_reparse_existing", return_value=(None, "none", "none", False, False)),
            patch("debate_v_majority.tools.rejudge_targets._extract_only_existing_result", return_value=expected) as p_extract,
        ):
            plan = rt._plan_target_row(
                dataset="gpqa",
                path=Path("/tmp/t.jsonl"),
                row=row,
                row_idx=0,
                orig_id=11,
                extract_only_existing=True,
                engine_mgr=MinimalManager(),
            )

        self.assertIsNotNone(plan.immediate_result)
        self.assertIsNone(plan.rerun_request)
        self.assertEqual(plan.immediate_result.action, "extract_only_failed")
        p_extract.assert_called_once()

    def test_plan_target_row_missing_model_raises_when_not_extract_only(self):
        row = self._base_row(orig_id=22)
        with (
            patch("debate_v_majority.tools.rejudge_targets._needs_retry", return_value=True),
            patch("debate_v_majority.tools.rejudge_targets._try_reparse_existing", return_value=(None, "none", "none", False, False)),
        ):
            with self.assertRaisesRegex(
                ValueError,
                r"/tmp/t\.jsonl: orig_id=22 missing judge_trace\.judge_model; cannot rerun judge\.",
            ):
                rt._plan_target_row(
                    dataset="gpqa",
                    path=Path("/tmp/t.jsonl"),
                    row=row,
                    row_idx=0,
                    orig_id=22,
                    extract_only_existing=False,
                    engine_mgr=MinimalManager(),
                )

    def test_apply_rerun_attempt_result_action_mapping(self):
        row = self._base_row(orig_id=5)
        req = rt.RerunRequest(
            path=Path("/tmp/t.jsonl"),
            orig_id=5,
            row_idx=0,
            row=row,
            before=None,
            model_name="m",
        )

        def fake_apply_answer_update(**kwargs):
            kwargs["row"]["final_judge_answer"] = kwargs["judged_answer"]

        with patch("debate_v_majority.tools.rejudge_targets._apply_answer_update", side_effect=fake_apply_answer_update):
            retry_attempt = rt.JudgeAttempt(
                judged_answer="B",
                raw_output="raw",
                retry_output="retry",
                parse_failed=False,
                used_fallback=False,
                retry_used=True,
                parse_mode="strict",
                parse_source="retry_strict",
                retry_reason="parse_none",
                finish_state="retry_parsed",
                raw_had_strict_final=False,
                retry_had_strict_final=True,
                judge_context=[{"role": "system", "content": "ctx"}],
            )
            retry_res = rt._apply_rerun_attempt_result(dataset="gpqa", req=req, attempt=retry_attempt)

            failed_attempt = rt.JudgeAttempt(
                judged_answer=None,
                raw_output="raw",
                retry_output="retry",
                parse_failed=True,
                used_fallback=False,
                retry_used=True,
                parse_mode="none",
                parse_source="none",
                retry_reason="parse_none",
                finish_state="retry_unparsed",
                raw_had_strict_final=False,
                retry_had_strict_final=False,
                judge_context=[{"role": "system", "content": "ctx"}],
            )
            failed_res = rt._apply_rerun_attempt_result(dataset="gpqa", req=req, attempt=failed_attempt)

        self.assertEqual(retry_res.action, "rerun_retry")
        self.assertIn("Retry was used.", retry_res.details)
        self.assertEqual(failed_res.action, "rerun_failed")
        self.assertIn("Still unparsable after retry.", failed_res.details)

    def test_run_judge_with_retry_batched_raw_and_extract_paths(self):
        row1 = self._base_row(orig_id=1)
        row1["judge_trace"] = {"judge_model": "m1"}
        row2 = self._base_row(orig_id=2)
        row2["judge_trace"] = {"judge_model": "m1"}

        req1 = rt.RerunRequest(
            path=Path("/tmp/t.jsonl"),
            orig_id=1,
            row_idx=0,
            row=row1,
            before=None,
            model_name="m1",
        )
        req2 = rt.RerunRequest(
            path=Path("/tmp/t.jsonl"),
            orig_id=2,
            row_idx=1,
            row=row2,
            before=None,
            model_name="m1",
        )

        engine = FakeEngine(
            outputs_by_call=[
                ["raw_good", "raw_bad"],
                ["retry_bad"],
                ["retry_bad2"],
                ["extract_good"],
            ]
        )
        manager = FakeEngineManager(engine)

        def parse_side_effect(*, dataset, text, raw_task, source_prefix, strict_enabled, recovery_enabled):
            t = str(text)
            if source_prefix == "raw" and not recovery_enabled:
                return rt.JudgeParseResult(answer=None, mode="none", source="none", strict_success=(t == "raw_good"))
            if source_prefix == "raw" and recovery_enabled:
                if t == "raw_good":
                    return rt.JudgeParseResult(answer="A", mode="strict", source="raw_strict", strict_success=True)
                return rt.JudgeParseResult(answer=None, mode="none", source="none", strict_success=False)
            if source_prefix == "retry" and not recovery_enabled:
                return rt.JudgeParseResult(answer=None, mode="none", source="none", strict_success=False)
            if source_prefix == "retry" and recovery_enabled:
                return rt.JudgeParseResult(answer=None, mode="none", source="none", strict_success=False)
            if source_prefix in ("extract_retry_only", "extract"):
                if t == "extract_good":
                    return rt.JudgeParseResult(
                        answer="D",
                        mode="strict",
                        source=f"{source_prefix}_strict",
                        strict_success=True,
                    )
                return rt.JudgeParseResult(answer=None, mode="none", source="none", strict_success=False)
            return rt.JudgeParseResult(answer=None, mode="none", source="none", strict_success=False)

        with (
            patch("debate_v_majority.tools.rejudge_targets._rebuild_judge_context", return_value=[{"role": "system", "content": "ctx"}]),
            patch("debate_v_majority.tools.rejudge_targets._parse_judge_text", side_effect=parse_side_effect),
        ):
            attempts = rt._run_judge_with_retry_batched(
                dataset="gpqa",
                requests=[req1, req2],
                engine_mgr=manager,
                batch_size=8,
            )

        a1 = attempts[(Path("/tmp/t.jsonl"), 1)]
        a2 = attempts[(Path("/tmp/t.jsonl"), 2)]
        self.assertEqual(a1.finish_state, "raw_parsed")
        self.assertFalse(a1.retry_used)
        self.assertEqual(a2.finish_state, "extract_parsed")
        self.assertTrue(a2.retry_used)
        self.assertEqual(a2.retry_reason, "parse_none_then_extract_retry_only")
        self.assertEqual(a2.judged_answer, "D")
        self.assertEqual(len(engine.calls), 4)
        retry_ctx_call = engine.calls[1][0][0]
        self.assertEqual(retry_ctx_call[-2]["role"], "assistant")
        self.assertEqual(retry_ctx_call[-2]["content"], "raw_bad")
        self.assertEqual(retry_ctx_call[-1]["role"], "user")
        self.assertIn("unparsable", retry_ctx_call[-1]["content"])

    def test_run_retry_passes_uses_strict_probe_before_retry2(self):
        row = self._base_row(orig_id=9)
        req = rt.RerunRequest(
            path=Path("/tmp/t.jsonl"),
            orig_id=9,
            row_idx=0,
            row=row,
            before=None,
            model_name="m1",
        )
        meta = rt.PreparedRerunMeta(
            req=req,
            raw_task=row["raw_task"],
            judge_context=[{"role": "system", "content": "ctx"}],
        )
        engine = FakeEngine(outputs_by_call=[["retry_recoverish"], ["retry_strict"]])

        def parse_side_effect(*, dataset, text, raw_task, source_prefix, strict_enabled, recovery_enabled):
            self.assertEqual(source_prefix, "retry")
            self.assertTrue(strict_enabled)
            self.assertFalse(recovery_enabled)
            return rt.JudgeParseResult(answer=None, mode="none", source="none", strict_success=False)

        with patch("debate_v_majority.tools.rejudge_targets._parse_judge_text", side_effect=parse_side_effect):
            out = rt._run_retry_passes_for_model(
                dataset="gpqa",
                engine=engine,
                metas=[meta],
                raw_outputs=["raw_bad"],
                retry_pending=[0],
                batch_size=8,
                retry_sampling={"max_tokens": 64},
            )

        self.assertEqual(out[0], "retry_strict")
        self.assertEqual(len(engine.calls), 2)

    def test_main_extract_only_missing_model_reports_failure_instead_of_aborting(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "targets.jsonl"
            row = self._base_row(orig_id=7)
            row["judge_trace"] = {}
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            forced = rt.RowUpdateResult(
                path=path,
                orig_id=7,
                action="extract_only_failed",
                before=None,
                after=None,
                changed=False,
                details="Missing judge_trace.judge_model; cannot run extraction-only pass.",
            )

            argv = [
                "debate_v_majority.tools.rejudge_targets.py",
                "--dataset",
                "gpqa",
                "--target",
                f"{path}:7",
                "--extract_only_existing",
                "--dry_run",
            ]
            out_buf = io.StringIO()
            err_buf = io.StringIO()

            with (
                patch("debate_v_majority.tools.rejudge_targets.EngineManager", MinimalManager),
                patch("debate_v_majority.tools.rejudge_targets._needs_retry", return_value=True),
                patch("debate_v_majority.tools.rejudge_targets._try_reparse_existing", return_value=(None, "none", "none", False, False)),
                patch("debate_v_majority.tools.rejudge_targets._extract_only_existing_result", return_value=forced),
                patch.object(sys, "argv", argv),
                redirect_stdout(out_buf),
                redirect_stderr(err_buf),
            ):
                rt.main()

            out = out_buf.getvalue()
            self.assertIn("Targets processed: 1", out)
            self.assertIn("- extract_only_failed: 1", out)
            self.assertIn(f"- {path} orig_id=7: extract_only_failed", out)

    def test_arg_parser_judge_max_tokens_default_is_none(self):
        ap = rt._build_arg_parser()
        args = ap.parse_args(["--dataset", "gpqa", "--target", "/tmp/t.jsonl:1"])
        self.assertIsNone(args.judge_max_tokens)


if __name__ == "__main__":
    unittest.main()
