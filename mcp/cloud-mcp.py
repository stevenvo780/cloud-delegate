#!/usr/bin/env python3
"""
cloud-mcp — puente MCP (stdio, sin dependencias) entre clientes y CLI de agentes locales.

Cualquier harness que hable MCP (Claude Code, Codex CLI, OpenCode, Gemini CLI, Antigravity `agy`)
carga este servidor y obtiene la tool `delegar_a_cloud`, que ejecuta la tarea en OTRO harness/CLI
con la autenticación ya configurada en esa máquina:

  claude/opus|sonnet|haiku|fable     -> `claude -p`   (CLI nativo)
  codex/gpt-5.6-sol|luna|terra       -> `codex exec`  (failover opcional entre CODEX_HOME)
  codex/gpt-5.3-codex-spark          -> `codex exec`
  gemini/flash|pro                   -> `agy -p`      (Antigravity)
  minimax/MiniMax-M3|M2.7            -> `opencode run` (MiniMax Token Plan)
  grok/grok-4.6|grok-4.5             -> `grok -p`     (Grok Build; alias grok/4.6|4.5)

Rutas NO enumeradas se rechazan (openrouter, groq, cerebras, nvidia, opencode-go, ollama...).

Nivel de acceso del delegado (`access`):
  text  (default) solo razona sobre el prompt; sin herramientas (o el mínimo del CLI: plan/read-only).
  read  puede leer el directorio `cwd` (Read/Glob/Grep, sandbox read-only, modo plan).
  write puede editar en `cwd` (bypass de permisos / workspace-write). Úsalo a conciencia.

Anti-recursión: cada salto incrementa CLOUD_OFFLOAD_DEPTH en el hijo; al llegar a
CLOUD_OFFLOAD_MAX_DEPTH (default 2) el servidor rechaza delegar otra vez.

Config por env vars (todas opcionales):
  CLOUD_OFFLOAD_MODEL        modelo por defecto (default gemini/flash)
  CLOUD_OFFLOAD_TIMEOUT_S    timeout por defecto en segundos (default 900, máx 1800)
  CLOUD_OFFLOAD_CODEX_HOMES  lista `a:b:c` de CODEX_HOME a probar en orden; ante 401/refresh
                             revocado/usage limit se salta al siguiente. Default: $CODEX_HOME si
                             está fijado; si no, ~/.codex.
  CLOUD_OFFLOAD_MAX_DEPTH    profundidad máxima de delegación encadenada (default 2)

Todo fallo se devuelve como `isError: true` con el motivo REAL (stderr redactado), nunca como
"(respuesta vacía)".
"""
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

VERSION = "2.0.0"
SERVER_NAME = "cloud-offload"

DEFAULT_MODEL = os.environ.get("CLOUD_OFFLOAD_MODEL") or "gemini/flash"
MAX_TIMEOUT = 1800.0
MIN_TIMEOUT = 30.0
MAX_PROMPT_CHARS = 4_000_000

MINIMAX_MODELS = frozenset(("minimax/MiniMax-M3", "minimax/MiniMax-M2.7"))
GEMINI_ALIASES = {
    "gemini/flash": "flash",
    "gemini/pro": "pro",
    "gemini/3.7-flash": "flash",
    "gemini/3.1-pro": "pro",
}
CLAUDE_MODELS = {
    "claude/opus": "opus",
    "claude/sonnet": "sonnet",
    "claude/haiku": "haiku",
    "claude/fable": "claude-fable-5-1",
}
CODEX_MODELS = {
    "codex/gpt-5.6-sol": "gpt-5.6-sol",
    "codex/gpt-5.6-luna": "gpt-5.6-luna",
    "codex/gpt-5.6-terra": "gpt-5.6-terra",
    "codex/gpt-5.3-codex-spark": "gpt-5.3-codex-spark",
}
GROK_MODELS = {
    "grok/grok-4.6": "grok-4.6",
    "grok/grok-4.5": "grok-4.5",
    "grok/4.6": "grok-4.6",
    "grok/4.5": "grok-4.5",
}
SUPPORTED_MODELS = tuple(sorted(
    set(MINIMAX_MODELS) | set(GEMINI_ALIASES) | set(CLAUDE_MODELS) | set(CODEX_MODELS) | set(GROK_MODELS)
))
ACCESS_LEVELS = ("text", "read", "write")
EFFORT_LEVELS = ("low", "medium", "high", "xhigh")

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_SECRET_PATTERNS = (
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{16,}"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
)
_CODEX_FAILOVER = re.compile(
    r"(?i)usage limit|hit your usage|401|unauthorized|refresh token|token_revoked|"
    r"refresh_token_invalidated|could not be refreshed|Missing bearer|log out and sign in"
)

# Antigravity rota su catálogo cada pocas semanas y retira modelos sin avisar. Estos son los
# nombres preferidos hoy; si el CLI los rechaza, _agy_resolve() recalcula consultando `agy models`.
_AGY_PREFERRED = {"pro": "Gemini 3.1 Pro (High)", "flash": "Gemini 3.7 Flash (Medium)"}


class DelegationError(Exception):
    """Fallo de una delegación; el mensaje ya viene redactado y listo para el cliente MCP."""


# ----------------------------------------------------------------------------- utilidades

def _depth():
    try:
        return int(os.environ.get("CLOUD_OFFLOAD_DEPTH", "0") or 0)
    except ValueError:
        return 0


def _max_depth():
    try:
        return int(os.environ.get("CLOUD_OFFLOAD_MAX_DEPTH", "2") or 2)
    except ValueError:
        return 2


def _default_timeout():
    try:
        return _clamp_timeout(float(os.environ.get("CLOUD_OFFLOAD_TIMEOUT_S", "900")))
    except (TypeError, ValueError):
        return 900.0


def _clamp_timeout(value):
    if value is None:
        return _default_timeout()
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise DelegationError(f"timeout_s inválido: {value!r}")
    return max(MIN_TIMEOUT, min(MAX_TIMEOUT, value))


def _redact(text, limit=900):
    out = _ANSI.sub("", text or "").replace("\x00", "")
    for pat in _SECRET_PATTERNS:
        out = pat.sub("[REDACTED]", out)
    out = "\n".join(l.rstrip() for l in out.splitlines() if l.strip())
    return out[-limit:] if len(out) > limit else out


def _clean(text, drop_substr=(), drop_prefix=(), drop_exact=()):
    out = _ANSI.sub("", text or "")
    keep = []
    for line in out.splitlines():
        s = line.strip()
        if any(sub in line for sub in drop_substr):
            continue
        if any(s.startswith(p) for p in drop_prefix):
            continue
        if s in drop_exact:
            continue
        keep.append(line.rstrip())
    return "\n".join(keep).strip()


def _base_env():
    env = dict(os.environ)
    env["CLOUD_OFFLOAD_DEPTH"] = str(_depth() + 1)
    return env


def _claude_env():
    """Aísla el Claude nativo de proxies/modelos/sesión heredados del orquestador (claude-gpt, etc.)."""
    env = {
        name: value for name, value in _base_env().items()
        if not name.startswith("ANTHROPIC_")
        and name != "CLAUDECODE"
        and not name.startswith("CLAUDE_CODE_")
    }
    return env


def _codex_homes():
    raw = os.environ.get("CLOUD_OFFLOAD_CODEX_HOMES")
    if raw:
        homes = [h for h in raw.split(":") if h]
    elif os.environ.get("CODEX_HOME"):
        homes = [os.environ["CODEX_HOME"]]
    else:
        homes = [os.path.expanduser("~/.codex")]
    seen, ordered = set(), []
    for h in homes:
        h = os.path.expanduser(h)
        if h not in seen:
            seen.add(h)
            ordered.append(h)
    return ordered


def _run_process(args, *, timeout, input=None, env=None, cwd=None):
    """Ejecuta un CLI en su propia sesión y mata el árbol entero si vence el timeout."""
    try:
        process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=cwd,
            start_new_session=True,
        )
    except FileNotFoundError:
        raise DelegationError(f"CLI no encontrado en PATH: {args[0]}")
    try:
        stdout, stderr = process.communicate(input=input, timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=2)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(args, timeout, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)


def _fail(cli, p, extra=""):
    detail = _redact((p.stderr or "") + "\n" + (p.stdout or ""))
    msg = f"{cli} terminó con código {p.returncode}"
    if extra:
        msg += f" — {extra}"
    if detail:
        msg += f"\n--- stderr/stdout ---\n{detail}"
    raise DelegationError(msg)


def _with_system(prompt, system):
    return f"<<SYS>>\n{system}\n<</SYS>>\n\n{prompt}" if system else prompt


# ----------------------------------------------------------------------------- rutas

def _route(model):
    if model in CLAUDE_MODELS:
        return "claude", CLAUDE_MODELS[model]
    if model in CODEX_MODELS:
        return "codex", CODEX_MODELS[model]
    if model in GEMINI_ALIASES:
        return "gemini", GEMINI_ALIASES[model]
    if model.startswith("gemini/") and ("(" in model or model[7:].startswith("Gemini ")):
        return "gemini", model.split("/", 1)[1]        # nombre nativo del catálogo de agy
    if model in MINIMAX_MODELS:
        return "minimax", model
    if model in GROK_MODELS:
        return "grok", GROK_MODELS[model]
    raise DelegationError(
        f"Modelo no permitido: '{model}'. Rutas habilitadas: {', '.join(SUPPORTED_MODELS)}. "
        "También se acepta gemini/<nombre exacto del catálogo `agy models`>."
    )


def _call_claude(prompt, system, alias, access, cwd, effort, timeout):
    args = [
        "claude", "-p",
        "--model", alias,
        "--output-format", "json",
        "--no-session-persistence",
        "--permission-mode", "bypassPermissions",
        "--strict-mcp-config",          # el hijo no carga MCPs (ni este) -> arranque rápido, sin bucles
    ]
    # Sin --bare: ese modo puede impedir leer la autenticación local del CLI.
    # --strict-mcp-config evita cargar MCPs en el hijo.
    if access == "text":
        args += ["--tools", ""]
    elif access == "read":
        args += ["--tools", "Read,Glob,Grep"]
    else:
        args += ["--tools", "default"]
    if system:
        args += ["--system-prompt", system]
    if effort:
        args += ["--effort", effort]
    p = _run_process(args, timeout=timeout, input=prompt, env=_claude_env(), cwd=cwd)
    text, meta = None, {}
    raw = (p.stdout or "").strip()
    try:
        data = json.loads(raw) if raw.startswith("{") else None
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        text = data.get("result")
        meta = {k: data.get(k) for k in ("duration_ms", "num_turns", "total_cost_usd", "session_id")
                if data.get(k) is not None}
        if data.get("is_error"):
            raise DelegationError(f"claude devolvió is_error: {_redact(str(text))}")
    if p.returncode != 0:
        _fail("claude", p)
    if text is None:
        text = _clean(p.stdout)
    if not text:
        _fail("claude", p, "respuesta vacía")
    return text, meta


def _codex_args(native, access, cwd, effort, out_path):
    args = [
        "codex", "exec",
        "--model", native,
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-user-config",         # sin MCPs/hooks del usuario: arranque limpio; auth sigue en CODEX_HOME
        "--color", "never",
        "-o", out_path,
        "-c", 'approval_policy="never"',
        "-s", "workspace-write" if access == "write" else "read-only",
    ]
    if effort:
        args += ["-c", f'model_reasoning_effort="{effort}"']
    if cwd:
        args += ["-C", cwd]
    args.append("-")                    # el prompt entra por stdin
    return args


def _call_codex(prompt, system, native, access, cwd, effort, timeout):
    full = _with_system(prompt, system)
    errors = []
    homes = _codex_homes()
    for index, home in enumerate(homes, 1):
        fd, out_path = tempfile.mkstemp(prefix="cloud-mcp-codex-", suffix=".txt")
        os.close(fd)
        try:
            env = _base_env()
            env["CODEX_HOME"] = home
            p = _run_process(_codex_args(native, access, cwd, effort, out_path), timeout=timeout,
                             input=full, env=env, cwd=cwd)
            try:
                with open(out_path, encoding="utf-8", errors="replace") as fh:
                    last = fh.read().strip()
            except OSError:
                last = ""
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass
        if p.returncode == 0 and last:
            return last, {"codex_home_index": index}
        detail = _redact((p.stderr or "") + "\n" + (p.stdout or ""), limit=600)
        errors.append(f"[CODEX_HOME candidato {index}] código {p.returncode}: {detail or '(sin salida)'}")
        if not _CODEX_FAILOVER.search(detail):
            break
    raise DelegationError("codex falló en todos los CODEX_HOME probados:\n" + "\n".join(errors))


def _agy_model(suffix):
    if suffix.startswith("Gemini ") or "(" in suffix:
        return suffix
    return _AGY_PREFERRED["pro" if "pro" in suffix.lower() else "flash"]


def _agy_catalog():
    try:
        p = _run_process(["agy", "models"], timeout=60)
    except (subprocess.TimeoutExpired, DelegationError):
        return []
    names = []
    for line in _clean(p.stdout).splitlines():
        parts = line.split("\t")
        if len(parts) == 2 and parts[1].strip():
            names.append(parts[1].strip())
    return names


def _agy_resolve(suffix):
    kind = "pro" if "pro" in suffix.lower() else "flash"
    wanted = "Pro" if kind == "pro" else "Flash"
    grade = "(High)" if kind == "pro" else "(Medium)"
    catalog = [n for n in _agy_catalog() if n.startswith("Gemini ") and wanted in n]
    if not catalog:
        return None
    for name in catalog:
        if name.endswith(grade):
            return name
    return catalog[0]


def _agy_args(full, name, access, effort, timeout, cwd=None):
    args = ["agy", "-p", full, "--model", name, "--output-format", "text",
            "--print-timeout", f"{int(timeout)}s"]
    if access == "text":
        args += ["--sandbox", "--mode", "plan"]
    elif access == "read":
        # En headless agy auto-deniega cualquier tool que pida permiso (hasta un `ls`) y devuelve
        # vacío; plan = solo lectura, y skip-permissions deja que esas lecturas corran.
        args += ["--mode", "plan", "--dangerously-skip-permissions"]
    else:
        args += ["--mode", "accept-edits", "--dangerously-skip-permissions"]
    if access != "text" and cwd:
        # agy trabaja sobre su "workspace", no sobre el cwd del proceso: hay que añadirlo explícitamente.
        args += ["--add-dir", cwd]
    if effort:
        args += ["--effort", "high" if effort == "xhigh" else effort]
    return args


def _call_agy(prompt, system, suffix, access, cwd, effort, timeout):
    full = _with_system(prompt, system)
    drop = ("Could not read directory", "Warning:", "trusted directory", "Ripgrep is not available")

    def run(name):
        p = _run_process(_agy_args(full, name, access, effort, timeout, cwd), timeout=timeout,
                         env=_base_env(), cwd=cwd)
        return _clean(p.stdout, drop_substr=drop), p, name

    out, p, used = run(_agy_model(suffix))
    if (not out or p.returncode != 0) and "invalid model selection" in (p.stderr or "").lower():
        alt = _agy_resolve(suffix)
        if alt and alt != used:
            out, p, used = run(alt)
    if p.returncode != 0:
        _fail("agy", p)
    if not out:
        _fail("agy", p, "respuesta vacía")
    return out, {"agy_model": used}


def _call_opencode(prompt, system, model, access, cwd, timeout):
    full = _with_system(prompt, system)
    args = ["opencode", "run", "--pure", "--model", model, "--format", "default"]
    if access == "write":
        args += ["--auto"]
    else:
        args += ["--agent", "plan"]
    if cwd:
        args += ["--dir", cwd]
    args.append("-")
    p = _run_process(args, timeout=timeout, input=full, env=_base_env(), cwd=cwd)
    # opencode imprime un banner "> build · <modelo>" (o "> plan · ...") antes de la respuesta.
    out = _clean(p.stdout, drop_prefix=("build ·", "plan ·", "> build ·", "> plan ·", "▣"),
                 drop_exact=(">",))
    if p.returncode != 0:
        _fail("opencode", p)
    if not out:
        _fail("opencode", p, "respuesta vacía")
    return out, {}


def _grok_args(native, prompt_path, access, cwd, effort):
    # El prompt va por archivo (--prompt-file): sin límite de argv y sin expansión de /comandos.
    args = ["grok", "--prompt-file", prompt_path, "--verbatim", "--output-format", "json",
            "-m", native, "--permission-mode", "bypassPermissions"]
    if access == "text":
        args += ["--tools", ""]                              # sin herramientas: solo razona
    elif access == "read":
        args += ["--tools", "read_file,grep,list_dir"]        # IDs internos de grok (headless)
    if cwd:
        args += ["--cwd", cwd]
    if effort:
        args += ["--reasoning-effort", effort]
    return args


def _call_grok(prompt, system, native, access, cwd, effort, timeout):
    full = _with_system(prompt, system)
    fd, prompt_path = tempfile.mkstemp(prefix="cloud-mcp-grok-", suffix=".md")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(full)
        p = _run_process(_grok_args(native, prompt_path, access, cwd, effort), timeout=timeout,
                         env=_base_env(), cwd=cwd)
    finally:
        try:
            os.unlink(prompt_path)
        except OSError:
            pass
    raw = (p.stdout or "").strip()
    data = None
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = None
    if p.returncode != 0:
        _fail("grok", p)
    if isinstance(data, dict):
        text = (data.get("text") or "").strip()
        meta = {k: data.get(k) for k in ("num_turns", "total_cost_usd", "sessionId", "stopReason")
                if data.get(k) is not None}
    else:
        text, meta = _clean(p.stdout), {}
    if not text:
        _fail("grok", p, "respuesta vacía")
    return text, meta


# ----------------------------------------------------------------------------- API de la tool

def call_cloud(prompt, system=None, model=None, effort=None, access=None, cwd=None, timeout_s=None):
    if _depth() >= _max_depth():
        raise DelegationError(
            f"Profundidad de delegación máxima alcanzada (CLOUD_OFFLOAD_DEPTH={_depth()}, "
            f"máx {_max_depth()}). Resolvé la tarea en este agente."
        )
    if not isinstance(prompt, str) or not prompt.strip():
        raise DelegationError("'prompt' debe ser un texto no vacío.")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise DelegationError(f"'prompt' excede {MAX_PROMPT_CHARS} caracteres.")
    model = model or DEFAULT_MODEL
    provider, native = _route(model)
    access = (access or "text").lower()
    if access not in ACCESS_LEVELS:
        raise DelegationError(f"'access' inválido: {access!r}. Valores: {', '.join(ACCESS_LEVELS)}.")
    if effort is not None:
        effort = str(effort).lower()
        if effort == "minimal":
            effort = "low"
        if effort not in EFFORT_LEVELS:
            raise DelegationError(f"'effort' inválido: {effort!r}. Valores: {', '.join(EFFORT_LEVELS)}.")
    cwd = os.path.abspath(os.path.expanduser(cwd)) if cwd else os.getcwd()
    if not os.path.isdir(cwd):
        raise DelegationError(f"'cwd' no es un directorio: {cwd}")
    timeout = _clamp_timeout(timeout_s)

    started = time.monotonic()
    try:
        if provider == "claude":
            text, meta = _call_claude(prompt, system, native, access, cwd, effort, timeout)
        elif provider == "codex":
            text, meta = _call_codex(prompt, system, native, access, cwd, effort, timeout)
        elif provider == "gemini":
            text, meta = _call_agy(prompt, system, native, access, cwd, effort, timeout)
        elif provider == "grok":
            text, meta = _call_grok(prompt, system, native, access, cwd, effort, timeout)
        else:
            text, meta = _call_opencode(prompt, system, native, access, cwd, timeout)
    except subprocess.TimeoutExpired as exc:
        raise DelegationError(
            f"{provider} superó el timeout de {int(timeout)} s (proceso terminado). "
            f"Salida parcial: {_redact(str(exc.output or ''), 400)}"
        )
    meta.update({
        "provider": provider,
        "model": model,
        "access": access,
        "duration_s": round(time.monotonic() - started, 1),
        "depth": _depth(),
    })
    return text, meta


def list_routes():
    rows = []
    for m in SUPPORTED_MODELS:
        provider, native = _route(m)
        cli = {"claude": "claude", "codex": "codex", "gemini": "agy", "minimax": "opencode", "grok": "grok"}[provider]
        rows.append({"model": m, "provider": provider, "cli": cli, "cli_available": shutil.which(cli) is not None,
                     "native": native})
    home_count = len(_codex_homes())
    return {
        "server": SERVER_NAME, "version": VERSION, "default_model": DEFAULT_MODEL,
        "default_timeout_s": _default_timeout(), "depth": _depth(), "max_depth": _max_depth(),
        "access_levels": list(ACCESS_LEVELS), "effort_levels": list(EFFORT_LEVELS),
        "codex_home_count": home_count, "routes": rows,
    }


TOOLS = [
    {
        "name": "delegar_a_cloud",
        "description": (
            "Delega una tarea a OTRO CLI/modelo configurado localmente y devuelve su respuesta completa. "
            "Rutas: claude/opus|sonnet|haiku|fable (CLI nativo), "
            "codex/gpt-5.6-sol|luna|terra y codex/gpt-5.3-codex-spark (Codex CLI), "
            "gemini/flash|pro (Antigravity agy), minimax/MiniMax-M3|M2.7 (OpenCode), "
            "grok/grok-4.6|grok-4.5 (Grok Build CLI). "
            f"Default: {DEFAULT_MODEL}. El delegado NO ve tu conversación: incluí TODO el contexto en "
            "'prompt'. 'access' controla qué puede tocar: text (solo razona; default), read (lee 'cwd'), "
            "write (edita en 'cwd'). Con access=read/write pasá 'cwd' al repo. Los errores llegan con el "
            "motivo real (401, usage limit, timeout...). Máximo 2 saltos encadenados."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Tarea completa y autocontenida."},
                "system": {"type": "string", "description": "Instrucción de sistema opcional (rol/estilo/restricciones)."},
                "model": {"type": "string", "description": "Ruta modelo. Ver descripción; también gemini/<nombre exacto de `agy models`>."},
                "effort": {"type": "string", "enum": list(EFFORT_LEVELS),
                           "description": "Esfuerzo de razonamiento: codex (model_reasoning_effort), claude (--effort), agy (--effort, xhigh→high) y grok (--reasoning-effort). Ignorado por minimax."},
                "access": {"type": "string", "enum": list(ACCESS_LEVELS),
                           "description": "text = sin herramientas (default); read = puede leer cwd; write = puede editar cwd."},
                "cwd": {"type": "string", "description": "Directorio de trabajo del delegado (default: el cwd del servidor MCP = tu proyecto)."},
                "timeout_s": {"type": "number", "description": f"Timeout en segundos (default {int(_default_timeout())}, máx {int(MAX_TIMEOUT)})."},
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "listar_modelos_cloud",
        "description": ("Lista las rutas modelo disponibles, qué CLI usa cada una y si ese CLI está en PATH, "
                        "la cantidad de CODEX_HOME candidatos y la profundidad de delegación actual. No consume cuota."),
        "inputSchema": {"type": "object", "properties": {}},
    },
]


# ----------------------------------------------------------------------------- servidor JSON-RPC

def send(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _tool_result(mid, text, structured=None, is_error=False):
    result = {"content": [{"type": "text", "text": text}]}
    if structured:
        result["structuredContent"] = structured
    if is_error:
        result["isError"] = True
    send({"jsonrpc": "2.0", "id": mid, "result": result})


def handle_tools_call(mid, params):
    name = params.get("name")
    args = params.get("arguments") or {}
    if not isinstance(args, dict):
        send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": "arguments debe ser un objeto"}})
        return
    if name == "listar_modelos_cloud":
        info = list_routes()
        _tool_result(mid, json.dumps(info, ensure_ascii=False, indent=2), info)
        return
    if name != "delegar_a_cloud":
        send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": f"unknown tool: {name}"}})
        return
    try:
        text, meta = call_cloud(
            args.get("prompt", ""), args.get("system"), args.get("model"), args.get("effort"),
            args.get("access"), args.get("cwd"), args.get("timeout_s"),
        )
    except DelegationError as exc:
        _tool_result(mid, f"Delegación fallida: {exc}", is_error=True)
    except Exception as exc:  # nunca dejar caer el servidor por una delegación
        _tool_result(mid, f"Error interno del puente cloud-offload: {type(exc).__name__}: {_redact(str(exc))}",
                     is_error=True)
    else:
        # Sin structuredContent: Claude Code muestra SOLO el structuredContent cuando existe y
        # esconde el texto de la respuesta. La metadata va como una línea final legible.
        _tool_result(mid, text + _meta_trailer(meta))


_TRAILER_KEYS = ("provider", "model", "access", "duration_s", "depth",
                 "codex_home_index", "agy_model", "num_turns", "total_cost_usd")


def _meta_trailer(meta):
    parts = [f"{k}={meta[k]}" for k in _TRAILER_KEYS if meta.get(k) is not None]
    return "\n\n[cloud-offload " + " ".join(parts) + "]"


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            send({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}})
            continue
        if not isinstance(msg, dict):
            continue
        mid = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": params.get("protocolVersion", "2025-06-18"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": VERSION},
            }})
        elif method in ("notifications/initialized", "initialized", "notifications/cancelled",
                        "notifications/roots/list_changed"):
            continue
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            handle_tools_call(mid, params)
        elif method == "ping":
            send({"jsonrpc": "2.0", "id": mid, "result": {}})
        elif mid is not None:
            send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}})


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
