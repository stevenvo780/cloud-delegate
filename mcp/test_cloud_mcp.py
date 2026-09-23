#!/usr/bin/env python3
"""Tests de cloud-mcp.py. Sin red ni cuota: mockean _run_process o usan CLIs de mentira en PATH.

Correr:  python3 mcp/test_cloud_mcp.py -v
"""
import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).with_name("cloud-mcp.py")
SPEC = importlib.util.spec_from_file_location("cloud_mcp", MODULE_PATH)
cloud_mcp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cloud_mcp)


def completed(stdout="OK", stderr="", returncode=0, args=None):
    return subprocess.CompletedProcess(args or [], returncode, stdout=stdout, stderr=stderr)


def claude_json(result="OK", is_error=False):
    return json.dumps({"type": "result", "result": result, "is_error": is_error,
                       "duration_ms": 1200, "num_turns": 1, "total_cost_usd": 0.0})


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {"CLOUD_OFFLOAD_DEPTH": "0", "CLOUD_OFFLOAD_MAX_DEPTH": "2"},
                                   clear=False)
        self.env.start()
        os.environ.pop("CLOUD_OFFLOAD_CODEX_HOMES", None)

    def tearDown(self):
        self.env.stop()

    # ---- claude
    @mock.patch.object(cloud_mcp, "_run_process")
    def test_claude_text_mode_uses_bare_no_tools_and_stdin(self, run):
        run.return_value = completed(stdout=claude_json("OK-CLAUDE"))
        text, meta = cloud_mcp.call_cloud("ping", model="claude/sonnet")
        self.assertEqual(text, "OK-CLAUDE")
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[:2], ["claude", "-p"])
        self.assertEqual(cmd[cmd.index("--model") + 1], "sonnet")
        self.assertNotIn("--bare", cmd)          # --bare nunca lee OAuth: rompe Claude Max
        self.assertEqual(cmd[cmd.index("--tools") + 1], "")
        self.assertIn("--strict-mcp-config", cmd)
        self.assertIn("--no-session-persistence", cmd)
        self.assertEqual(run.call_args.kwargs["input"], "ping")
        self.assertEqual(meta["provider"], "claude")
        self.assertEqual(meta["access"], "text")
        self.assertEqual(meta["num_turns"], 1)

    @mock.patch.object(cloud_mcp, "_run_process")
    def test_claude_read_and_write_modes(self, run):
        run.return_value = completed(stdout=claude_json())
        cloud_mcp.call_cloud("ping", model="claude/opus", access="read")
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[cmd.index("--tools") + 1], "Read,Glob,Grep")
        self.assertNotIn("--bare", cmd)
        cloud_mcp.call_cloud("ping", model="claude/opus", access="write")
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[cmd.index("--tools") + 1], "default")
        self.assertNotIn("--bare", cmd)
        self.assertEqual(cmd[cmd.index("--permission-mode") + 1], "bypassPermissions")

    @mock.patch.object(cloud_mcp, "_run_process")
    def test_claude_drops_parent_proxy_and_session_env_and_bumps_depth(self, run):
        run.return_value = completed(stdout=claude_json())
        inherited = {
            "ANTHROPIC_BASE_URL": "http://localhost:18765", "ANTHROPIC_API_KEY": "proxy-key",
            "ANTHROPIC_MODEL": "gpt-5.6-sol[1m]", "CLAUDECODE": "1", "CLAUDE_CODE_ENTRYPOINT": "cli",
            "HARNESS_TEST_KEEP": "yes", "CLOUD_OFFLOAD_DEPTH": "0",
        }
        with mock.patch.dict(os.environ, inherited):
            cloud_mcp.call_cloud("ping", model="claude/opus")
        child_env = run.call_args.kwargs["env"]
        self.assertEqual(child_env["HARNESS_TEST_KEEP"], "yes")
        self.assertEqual(child_env["CLOUD_OFFLOAD_DEPTH"], "1")
        for name in inherited:
            if name not in ("HARNESS_TEST_KEEP", "CLOUD_OFFLOAD_DEPTH"):
                self.assertNotIn(name, child_env)

    @mock.patch.object(cloud_mcp, "_run_process")
    def test_claude_system_prompt_effort_and_fable_id(self, run):
        run.return_value = completed(stdout=claude_json())
        cloud_mcp.call_cloud("payload", system="sos revisor", model="claude/fable", effort="high")
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[cmd.index("--system-prompt") + 1], "sos revisor")
        self.assertEqual(cmd[cmd.index("--effort") + 1], "high")
        self.assertEqual(cmd[cmd.index("--model") + 1], "claude-fable-5-1")

    @mock.patch.object(cloud_mcp, "_run_process")
    def test_claude_is_error_json_surfaces_reason(self, run):
        run.return_value = completed(stdout=claude_json("Not logged in", is_error=True))
        with self.assertRaises(cloud_mcp.DelegationError) as ctx:
            cloud_mcp.call_cloud("ping", model="claude/sonnet")
        self.assertIn("Not logged in", str(ctx.exception))

    @mock.patch.object(cloud_mcp, "_run_process")
    def test_claude_nonzero_exit_surfaces_stderr_not_empty_marker(self, run):
        run.return_value = completed(stdout="", stderr="Invalid API key · Please run /login", returncode=1)
        with self.assertRaises(cloud_mcp.DelegationError) as ctx:
            cloud_mcp.call_cloud("ping", model="claude/sonnet")
        self.assertIn("Invalid API key", str(ctx.exception))
        self.assertNotIn("respuesta vacía", str(ctx.exception))

    # ---- codex
    def _codex_side_effect(self, outcomes):
        """outcomes: lista de (returncode, stderr, last_message|None) por CODEX_HOME probado."""
        calls = []

        def fake(args, *, timeout, input=None, env=None, cwd=None):
            rc, stderr, last = outcomes[len(calls)]
            calls.append((args, env))
            out_path = args[args.index("-o") + 1]
            if last is not None:
                Path(out_path).write_text(last, encoding="utf-8")
            return completed(stdout="codex\nOK\ntokens used\n3.175\n", stderr=stderr, returncode=rc, args=args)
        return fake, calls

    def test_codex_uses_last_message_file_stdin_and_read_only_sandbox(self):
        fake, calls = self._codex_side_effect([(0, "", "OK-CODEX")])
        with mock.patch.object(cloud_mcp, "_run_process", side_effect=fake), \
             mock.patch.dict(os.environ, {"CLOUD_OFFLOAD_CODEX_HOMES": "/tmp/h1"}):
            text, meta = cloud_mcp.call_cloud("ping", model="codex/gpt-5.6-terra", effort="high")
        self.assertEqual(text, "OK-CODEX")
        args, env = calls[0]
        self.assertEqual(args[:4], ["codex", "exec", "--model", "gpt-5.6-terra"])
        self.assertIn('model_reasoning_effort="high"', args)
        self.assertEqual(args[args.index("-s") + 1], "read-only")
        self.assertIn("--ephemeral", args)
        self.assertIn("--ignore-user-config", args)
        self.assertEqual(args[-1], "-")
        self.assertEqual(env["CODEX_HOME"], "/tmp/h1")
        self.assertEqual(meta["codex_home_index"], 1)

    def test_codex_write_access_uses_workspace_write(self):
        fake, calls = self._codex_side_effect([(0, "", "hecho")])
        with mock.patch.object(cloud_mcp, "_run_process", side_effect=fake), \
             mock.patch.dict(os.environ, {"CLOUD_OFFLOAD_CODEX_HOMES": "/tmp/h1"}):
            cloud_mcp.call_cloud("ping", model="codex/gpt-5.6-sol", access="write")
        args, _ = calls[0]
        self.assertEqual(args[args.index("-s") + 1], "workspace-write")

    def test_codex_fails_over_to_next_home_on_usage_limit_or_401(self):
        fake, calls = self._codex_side_effect([
            (1, "ERROR: You've hit your usage limit. try again at Sep 10th", None),
            (1, "HTTP 401: refresh_token_invalidated", None),
            (0, "", "OK-3"),
        ])
        with mock.patch.object(cloud_mcp, "_run_process", side_effect=fake), \
             mock.patch.dict(os.environ, {"CLOUD_OFFLOAD_CODEX_HOMES": "/tmp/a:/tmp/b:/tmp/c"}):
            text, meta = cloud_mcp.call_cloud("ping", model="codex/gpt-5.6-sol")
        self.assertEqual(text, "OK-3")
        self.assertEqual([env["CODEX_HOME"] for _, env in calls], ["/tmp/a", "/tmp/b", "/tmp/c"])
        self.assertEqual(meta["codex_home_index"], 3)

    def test_codex_does_not_fail_over_on_unrelated_error_and_reports_it(self):
        fake, calls = self._codex_side_effect([(2, "error: unexpected argument '--zzz'", None), (0, "", "no")])
        with mock.patch.object(cloud_mcp, "_run_process", side_effect=fake), \
             mock.patch.dict(os.environ, {"CLOUD_OFFLOAD_CODEX_HOMES": "/tmp/a:/tmp/b"}):
            with self.assertRaises(cloud_mcp.DelegationError) as ctx:
                cloud_mcp.call_cloud("ping", model="codex/gpt-5.6-sol")
        self.assertEqual(len(calls), 1)
        self.assertIn("unexpected argument", str(ctx.exception))
        self.assertIn("CODEX_HOME candidato 1", str(ctx.exception))
        self.assertNotIn("/tmp/a", str(ctx.exception))

    def test_codex_all_homes_exhausted_lists_every_reason(self):
        fake, calls = self._codex_side_effect([
            (1, "usage limit", None), (1, "token_revoked", None)])
        with mock.patch.object(cloud_mcp, "_run_process", side_effect=fake), \
             mock.patch.dict(os.environ, {"CLOUD_OFFLOAD_CODEX_HOMES": "/tmp/a:/tmp/b"}):
            with self.assertRaises(cloud_mcp.DelegationError) as ctx:
                cloud_mcp.call_cloud("ping", model="codex/gpt-5.6-sol")
        msg = str(ctx.exception)
        self.assertIn("usage limit", msg)
        self.assertIn("token_revoked", msg)

    def test_codex_home_defaults(self):
        with mock.patch.dict(os.environ, {"CLOUD_OFFLOAD_CODEX_HOMES": "", "CODEX_HOME": "/x/.codex-kant"}):
            os.environ.pop("CLOUD_OFFLOAD_CODEX_HOMES")
            self.assertEqual(cloud_mcp._codex_homes(), ["/x/.codex-kant"])
        with mock.patch.dict(os.environ, {"CLOUD_OFFLOAD_CODEX_HOMES": "/a:/a:~/.codex"}):
            homes = cloud_mcp._codex_homes()
            self.assertEqual(homes[0], "/a")
            self.assertEqual(len(homes), 2)

    # ---- gemini / agy
    @mock.patch.object(cloud_mcp, "_run_process")
    def test_gemini_flash_uses_agy_plan_sandbox_and_print_timeout(self, run):
        run.return_value = completed(stdout="OK-AGY")
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "must-not-switch-cli"}):
            text, meta = cloud_mcp.call_cloud("ping", model="gemini/flash", timeout_s=120)
        self.assertEqual(text, "OK-AGY")
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[0], "agy")
        self.assertEqual(cmd[cmd.index("--model") + 1], "Gemini 3.7 Flash (Medium)")
        self.assertIn("--sandbox", cmd)
        self.assertEqual(cmd[cmd.index("--mode") + 1], "plan")
        self.assertEqual(cmd[cmd.index("--print-timeout") + 1], "120s")
        self.assertEqual(meta["agy_model"], "Gemini 3.7 Flash (Medium)")

    @mock.patch.object(cloud_mcp, "_run_process")
    def test_gemini_aliases_and_native_passthrough(self, run):
        run.return_value = completed(stdout="OK")
        cloud_mcp.call_cloud("ping", model="gemini/3.1-pro")
        self.assertEqual(run.call_args.args[0][run.call_args.args[0].index("--model") + 1], "Gemini 3.1 Pro (High)")
        cloud_mcp.call_cloud("ping", model="gemini/Gemini 3.8 Flash (High)")
        self.assertEqual(run.call_args.args[0][run.call_args.args[0].index("--model") + 1], "Gemini 3.8 Flash (High)")

    @mock.patch.object(cloud_mcp, "_run_process")
    def test_gemini_read_mode_skips_permission_prompts(self, run):
        run.return_value = completed(stdout="OK")
        with tempfile.TemporaryDirectory() as d:
            cloud_mcp.call_cloud("ping", model="gemini/flash", access="read", cwd=d)
            cmd = run.call_args.args[0]
            self.assertEqual(cmd[cmd.index("--add-dir") + 1], d)   # agy trabaja sobre su workspace
        self.assertEqual(cmd[cmd.index("--mode") + 1], "plan")
        self.assertIn("--dangerously-skip-permissions", cmd)   # headless auto-deniega tools si no
        self.assertNotIn("--sandbox", cmd)

    @mock.patch.object(cloud_mcp, "_run_process")
    def test_gemini_write_mode_and_effort_mapping(self, run):
        run.return_value = completed(stdout="OK")
        cloud_mcp.call_cloud("ping", model="gemini/pro", access="write", effort="xhigh")
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[cmd.index("--mode") + 1], "accept-edits")
        self.assertIn("--dangerously-skip-permissions", cmd)
        self.assertEqual(cmd[cmd.index("--effort") + 1], "high")
        self.assertNotIn("--sandbox", cmd)

    def test_gemini_retired_model_is_resolved_from_catalog(self):
        calls = []

        def fake(args, *, timeout, input=None, env=None, cwd=None):
            calls.append(args)
            if args[:2] == ["agy", "models"]:
                return completed(stdout="gemini-3.8-flash-high\tGemini 3.8 Flash (High)\n"
                                        "gemini-3.8-flash-medium\tGemini 3.8 Flash (Medium)\n"
                                        "gemini-3.1-pro-high\tGemini 3.1 Pro (High)\n")
            model = args[args.index("--model") + 1]
            if model == "Gemini 3.7 Flash (Medium)":
                return completed(stdout="", stderr="Error: invalid model selection", returncode=1)
            return completed(stdout="OK-NEW")
        with mock.patch.object(cloud_mcp, "_run_process", side_effect=fake):
            text, meta = cloud_mcp.call_cloud("ping", model="gemini/flash")
        self.assertEqual(text, "OK-NEW")
        self.assertEqual(meta["agy_model"], "Gemini 3.8 Flash (Medium)")

    # ---- minimax / opencode
    @mock.patch.object(cloud_mcp, "_run_process")
    def test_minimax_uses_opencode_plan_agent_pure_and_stdin(self, run):
        run.return_value = completed(stdout="\x1b[0m\n> build · MiniMax-M3\n\x1b[0m\nOK-MINIMAX\n")
        text, meta = cloud_mcp.call_cloud("ping", system="sos breve", model="minimax/MiniMax-M3")
        self.assertEqual(text, "OK-MINIMAX")
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[:2], ["opencode", "run"])
        self.assertIn("--pure", cmd)
        self.assertEqual(cmd[cmd.index("--model") + 1], "minimax/MiniMax-M3")
        self.assertEqual(cmd[cmd.index("--agent") + 1], "plan")
        self.assertEqual(cmd[-1], "-")
        self.assertTrue(run.call_args.kwargs["input"].startswith("<<SYS>>\nsos breve\n<</SYS>>\n\nping"))

    @mock.patch.object(cloud_mcp, "_run_process")
    def test_minimax_write_uses_auto(self, run):
        run.return_value = completed(stdout="ok")
        cloud_mcp.call_cloud("ping", model="minimax/MiniMax-M2.7", access="write")
        cmd = run.call_args.args[0]
        self.assertIn("--auto", cmd)
        self.assertNotIn("--agent", cmd)

    # ---- grok
    def _grok_side_effect(self, payload, calls):
        def fake(args, *, timeout, input=None, env=None, cwd=None):
            prompt_path = args[args.index("--prompt-file") + 1]
            calls.append((args, Path(prompt_path).read_text(encoding="utf-8"), env, cwd))
            return completed(stdout=json.dumps(payload), args=args)
        return fake

    def test_grok_text_uses_prompt_file_json_no_tools_and_cleans_up(self):
        calls = []
        payload = {"text": "OK-GROK", "num_turns": 1, "total_cost_usd": 0.0065, "sessionId": "s1", "stopReason": "end_turn"}
        with mock.patch.object(cloud_mcp, "_run_process", side_effect=self._grok_side_effect(payload, calls)):
            text, meta = cloud_mcp.call_cloud("ping", system="sos breve", model="grok/grok-4.6", effort="high")
        self.assertEqual(text, "OK-GROK")
        args, prompt_written, env, _ = calls[0]
        self.assertEqual(args[0], "grok")
        self.assertEqual(prompt_written, "<<SYS>>\nsos breve\n<</SYS>>\n\nping")
        self.assertFalse(Path(args[args.index("--prompt-file") + 1]).exists())   # temp borrado
        self.assertIn("--verbatim", args)
        self.assertEqual(args[args.index("--output-format") + 1], "json")
        self.assertEqual(args[args.index("-m") + 1], "grok-4.6")
        self.assertEqual(args[args.index("--tools") + 1], "")
        self.assertEqual(args[args.index("--permission-mode") + 1], "bypassPermissions")
        self.assertEqual(args[args.index("--reasoning-effort") + 1], "high")
        self.assertEqual(env["CLOUD_OFFLOAD_DEPTH"], "1")
        self.assertEqual(meta["num_turns"], 1)
        self.assertEqual(meta["provider"], "grok")

    def test_grok_read_write_aliases_and_cwd(self):
        calls = []
        payload = {"text": "ok"}
        with mock.patch.object(cloud_mcp, "_run_process", side_effect=self._grok_side_effect(payload, calls)), \
             tempfile.TemporaryDirectory() as d:
            cloud_mcp.call_cloud("ping", model="grok/4.5", access="read", cwd=d)
            args = calls[-1][0]
            self.assertEqual(args[args.index("-m") + 1], "grok-4.5")
            self.assertEqual(args[args.index("--tools") + 1], "read_file,grep,list_dir")
            self.assertEqual(args[args.index("--cwd") + 1], d)
            cloud_mcp.call_cloud("ping", model="grok/grok-4.6", access="write", cwd=d)
            args = calls[-1][0]
            self.assertNotIn("--tools", args)
            self.assertEqual(args[args.index("--permission-mode") + 1], "bypassPermissions")

    def test_grok_nonzero_exit_and_empty_text_are_errors(self):
        def failing(args, **kw):
            return completed(stdout="", stderr="error: not logged in. Run `grok login`", returncode=1, args=args)
        with mock.patch.object(cloud_mcp, "_run_process", side_effect=failing):
            with self.assertRaises(cloud_mcp.DelegationError) as ctx:
                cloud_mcp.call_cloud("ping", model="grok/grok-4.6")
        self.assertIn("grok login", str(ctx.exception))
        calls = []
        with mock.patch.object(cloud_mcp, "_run_process", side_effect=self._grok_side_effect({"text": ""}, calls)):
            with self.assertRaises(cloud_mcp.DelegationError) as ctx:
                cloud_mcp.call_cloud("ping", model="grok/grok-4.6")
        self.assertIn("respuesta vacía", str(ctx.exception))

    # ---- validaciones comunes
    @mock.patch.object(cloud_mcp, "_run_process")
    def test_unknown_models_rejected_before_spawn(self, run):
        for bad in ("codex/gpt-9-unknown", "minimax/not-a-plan-model", "claude/opus-5",
                    "openrouter/x", "groq/llama", "gemini/turbo", "ollama/qwen"):
            with self.assertRaises(cloud_mcp.DelegationError, msg=bad) as ctx:
                cloud_mcp.call_cloud("ping", model=bad)
            self.assertIn("Modelo no permitido", str(ctx.exception))
        run.assert_not_called()

    @mock.patch.object(cloud_mcp, "_run_process")
    def test_depth_guard_blocks_runaway_chains(self, run):
        with mock.patch.dict(os.environ, {"CLOUD_OFFLOAD_DEPTH": "2", "CLOUD_OFFLOAD_MAX_DEPTH": "2"}):
            with self.assertRaises(cloud_mcp.DelegationError) as ctx:
                cloud_mcp.call_cloud("ping", model="claude/sonnet")
        self.assertIn("Profundidad", str(ctx.exception))
        run.assert_not_called()

    @mock.patch.object(cloud_mcp, "_run_process")
    def test_invalid_access_effort_cwd_and_empty_prompt(self, run):
        with self.assertRaises(cloud_mcp.DelegationError):
            cloud_mcp.call_cloud("ping", model="claude/sonnet", access="root")
        with self.assertRaises(cloud_mcp.DelegationError):
            cloud_mcp.call_cloud("ping", model="codex/gpt-5.6-sol", effort="ultra")
        with self.assertRaises(cloud_mcp.DelegationError):
            cloud_mcp.call_cloud("ping", model="claude/sonnet", cwd="/definitely/not/here")
        with self.assertRaises(cloud_mcp.DelegationError):
            cloud_mcp.call_cloud("   ", model="claude/sonnet")
        run.assert_not_called()

    @mock.patch.object(cloud_mcp, "_run_process")
    def test_cwd_is_passed_to_child_and_timeout_clamped(self, run):
        run.return_value = completed(stdout=claude_json())
        with tempfile.TemporaryDirectory() as d:
            _, meta = cloud_mcp.call_cloud("ping", model="claude/sonnet", cwd=d, timeout_s=99999)
            self.assertEqual(run.call_args.kwargs["cwd"], os.path.realpath(d) if os.path.realpath(d) == d else d)
            self.assertEqual(run.call_args.kwargs["timeout"], cloud_mcp.MAX_TIMEOUT)
            self.assertNotIn("cwd", meta)

    def test_timeout_message_is_explicit(self):
        def boom(args, **kw):
            raise subprocess.TimeoutExpired(args, kw["timeout"], output="parcial")
        with mock.patch.object(cloud_mcp, "_run_process", side_effect=boom):
            with self.assertRaises(cloud_mcp.DelegationError) as ctx:
                cloud_mcp.call_cloud("ping", model="minimax/MiniMax-M3", timeout_s=30)
        self.assertIn("timeout de 30", str(ctx.exception))

    def test_redact_hides_tokens(self):
        api_example = "sk-" + "A" * 24
        github_example = "ghp_" + "B" * 24
        raw = f"Authorization: Bearer abcdefghijklmnop123 key {api_example} {github_example}"
        red = cloud_mcp._redact(raw)
        self.assertNotIn("abcdefghijklmnop123", red)
        self.assertNotIn(api_example, red)
        self.assertNotIn(github_example, red)
        self.assertIn("[REDACTED]", red)

    def test_list_routes_covers_every_supported_model(self):
        info = cloud_mcp.list_routes()
        self.assertEqual({r["model"] for r in info["routes"]}, set(cloud_mcp.SUPPORTED_MODELS))
        self.assertEqual(info["version"], cloud_mcp.VERSION)
        self.assertNotIn("codex_homes", info)
        self.assertNotIn("cwd", info)
        self.assertTrue(all("cli_path" not in route for route in info["routes"]))
        self.assertTrue(all(isinstance(route["cli_available"], bool) for route in info["routes"]))


class RunProcessTests(unittest.TestCase):
    def test_timeout_terminates_process_tree_without_pipe_hang(self):
        with tempfile.NamedTemporaryFile(delete=False) as pid_file:
            pid_path = pid_file.name
        script = textwrap.dedent("""
            import os, sys, time
            pid_path = sys.argv[1]
            child = os.fork()
            if child == 0:
                with open(pid_path, "w") as handle:
                    handle.write(str(os.getpid()))
                time.sleep(30)
            else:
                time.sleep(30)
        """)
        started = time.monotonic()
        try:
            with self.assertRaises(subprocess.TimeoutExpired):
                cloud_mcp._run_process([sys.executable, "-c", script, pid_path], timeout=0.3)
            self.assertLess(time.monotonic() - started, 3.0)
            grandchild_pid = int(Path(pid_path).read_text())
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                stat_path = Path(f"/proc/{grandchild_pid}/stat")
                if not stat_path.exists() or stat_path.read_text().split()[2] == "Z":
                    break
                time.sleep(0.05)
            else:
                self.fail("grandchild remained running after timeout")
        finally:
            Path(pid_path).unlink(missing_ok=True)

    def test_missing_cli_is_a_clear_error(self):
        with self.assertRaises(cloud_mcp.DelegationError) as ctx:
            cloud_mcp._run_process(["definitely-not-a-cli-xyz"], timeout=5)
        self.assertIn("CLI no encontrado", str(ctx.exception))


class StdioIntegrationTests(unittest.TestCase):
    """Levanta el servidor real por stdio con CLIs de mentira en PATH: prueba el protocolo de punta a punta."""

    STUBS = {
        "claude": textwrap.dedent("""\
            #!/usr/bin/env python3
            import json, sys
            prompt = sys.stdin.read()
            args = sys.argv[1:]
            model = args[args.index("--model") + 1]
            print(json.dumps({"type": "result", "result": f"[claude:{model}] {prompt.strip()}",
                              "is_error": False, "duration_ms": 5, "num_turns": 1}))
        """),
        "codex": textwrap.dedent("""\
            #!/usr/bin/env python3
            import os, sys
            args = sys.argv[1:]
            prompt = sys.stdin.read()
            home = os.environ.get("CODEX_HOME", "")
            if home.endswith("dead"):
                sys.stderr.write("ERROR: You've hit your usage limit.\\n"); sys.exit(1)
            out = args[args.index("-o") + 1]
            open(out, "w").write(f"[codex:{args[args.index('--model') + 1]}@{os.path.basename(home)}] {prompt.strip()}")
            print("codex\\nignored stdout\\ntokens used\\n42")
        """),
        "agy": textwrap.dedent("""\
            #!/usr/bin/env python3
            import sys
            args = sys.argv[1:]
            if args[:1] == ["models"]:
                print("gemini-3.7-flash-medium\\tGemini 3.7 Flash (Medium)"); sys.exit(0)
            print("Warning: something\\n[agy:%s] %s" % (args[args.index("--model") + 1], args[args.index("-p") + 1]))
        """),
        "opencode": textwrap.dedent("""\
            #!/usr/bin/env python3
            import sys
            prompt = sys.stdin.read()
            print("> build · MiniMax-M3\\n[opencode] " + prompt.strip())
        """),
        "grok": textwrap.dedent("""\
            #!/usr/bin/env python3
            import json, sys
            args = sys.argv[1:]
            prompt = open(args[args.index("--prompt-file") + 1]).read()
            print(json.dumps({"text": f"[grok:{args[args.index('-m') + 1]}] {prompt.strip()}",
                              "num_turns": 1, "total_cost_usd": 0.001, "stopReason": "end_turn"}, indent=2))
        """),
    }

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.bin = Path(cls.tmp.name) / "bin"
        cls.bin.mkdir()
        for name, body in cls.STUBS.items():
            p = cls.bin / name
            p.write_text(body)
            p.chmod(p.stat().st_mode | stat.S_IXUSR)
        (Path(cls.tmp.name) / "dead").mkdir()
        (Path(cls.tmp.name) / "alive").mkdir()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def rpc(self, calls, env_extra=None):
        env = dict(os.environ)
        env["PATH"] = f"{self.bin}{os.pathsep}{env.get('PATH', '')}"
        env["CLOUD_OFFLOAD_DEPTH"] = "0"
        env["CLOUD_OFFLOAD_CODEX_HOMES"] = f"{self.tmp.name}/dead:{self.tmp.name}/alive"
        env.update(env_extra or {})
        msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize",
                 "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}]
        for i, (name, args) in enumerate(calls, start=10):
            msgs.append({"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": name, "arguments": args}})
        p = subprocess.run([sys.executable, str(MODULE_PATH)], input="".join(json.dumps(m) + "\n" for m in msgs),
                           capture_output=True, text=True, timeout=60, env=env, cwd=self.tmp.name)
        self.assertEqual(p.returncode, 0, p.stderr)
        out = {}
        for line in p.stdout.splitlines():
            d = json.loads(line)          # cada línea de stdout DEBE ser JSON-RPC válido
            out[d.get("id")] = d
        return out

    def test_protocol_and_every_route_end_to_end(self):
        out = self.rpc([
            ("delegar_a_cloud", {"prompt": "hola", "model": "claude/sonnet"}),
            ("delegar_a_cloud", {"prompt": "hola", "model": "codex/gpt-5.6-sol"}),
            ("delegar_a_cloud", {"prompt": "hola", "model": "gemini/flash"}),
            ("delegar_a_cloud", {"prompt": "hola", "model": "minimax/MiniMax-M3"}),
            ("delegar_a_cloud", {"prompt": "hola", "model": "groq/nope"}),
            ("listar_modelos_cloud", {}),
            ("delegar_a_cloud", {"prompt": "sobre /workspace/x.py y ~/.ssh: resumí", "model": "claude/sonnet"}),
            ("delegar_a_cloud", {"prompt": "hola", "model": "grok/4.6"}),
        ])
        self.assertEqual(out[17]["result"]["content"][0]["text"].split("\n\n[cloud-offload ")[0], "[grok:grok-4.6] hola")
        self.assertIn("provider=grok model=grok/4.6", out[17]["result"]["content"][0]["text"])
        self.assertEqual(out[1]["result"]["serverInfo"]["name"], "cloud-offload")
        self.assertEqual({t["name"] for t in out[2]["result"]["tools"]}, {"delegar_a_cloud", "listar_modelos_cloud"})

        def text(i):
            return out[i]["result"]["content"][0]["text"]

        def body(i):   # respuesta del delegado sin la línea final de metadata
            t = text(i)
            self.assertIn("\n\n[cloud-offload provider=", t)
            return t.split("\n\n[cloud-offload ")[0]
        self.assertEqual(body(10), "[claude:sonnet] hola")
        self.assertEqual(body(11), "[codex:gpt-5.6-sol@alive] hola")        # saltó el home muerto
        self.assertIn("codex_home_index=2", text(11))
        self.assertNotIn(self.tmp.name, text(11))
        self.assertEqual(body(12), "[agy:Gemini 3.7 Flash (Medium)] hola")   # Warning: filtrado
        self.assertEqual(body(13), "[opencode] hola")                        # banner build · filtrado
        self.assertTrue(out[14]["result"].get("isError"))
        self.assertIn("Modelo no permitido", text(14))
        self.assertEqual(out[15]["result"]["structuredContent"]["version"], cloud_mcp.VERSION)
        # rutas privadas en el prompt NO se rechazan: el delegado es un harness del operador
        self.assertEqual(body(16), "[claude:sonnet] sobre /workspace/x.py y ~/.ssh: resumí")
        for i in (10, 11, 12, 13):
            # Claude Code muestra SOLO structuredContent si existe y esconde el texto: no debe haber.
            self.assertNotIn("structuredContent", out[i]["result"])
            self.assertNotIn("isError", out[i]["result"])
        self.assertIn("provider=claude model=claude/sonnet access=text", text(10))

    def test_depth_limit_reached_is_an_error_not_a_crash(self):
        out = self.rpc([("delegar_a_cloud", {"prompt": "x", "model": "claude/sonnet"})],
                       env_extra={"CLOUD_OFFLOAD_DEPTH": "2"})
        self.assertTrue(out[10]["result"]["isError"])
        self.assertIn("Profundidad", out[10]["result"]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
