#!/usr/bin/env python3
"""Install data-analyst-mcp into one of six MCP clients.

Supported clients (one positional argument):

  claude-code     project-scoped `.mcp.json` at the repo root
  claude-desktop  user-global config; path auto-detected per OS
  codex           `~/.codex/config.toml` (Codex CLI / IDE extension)
  cursor          project-scoped `.cursor/mcp.json` at the repo root
  opencode        project-scoped `opencode.json` at the repo root
  antigravity     prints a JSON snippet to paste into the editor UI
  all             every client above

Default command wires `uv --directory <repo> run data-analyst-mcp`, so the
client always boots the in-tree code. Pass `--installed` to switch to
`uvx --from git+...@<tag> data-analyst-mcp`, which clones a pinned tag
into an isolated environment per run.

Existing servers in the target config are preserved — only the
`data-analyst` entry (or whatever you pass via `--name`) is added or
replaced. Pass `--dry-run` to print the planned write without touching
the file.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NAME = "data-analyst"
GIT_REF = "v0.1.0"
GIT_URL = "git+https://github.com/olegtyshcneko/data-analyst-mcp"

CLIENTS = (
    "claude-code",
    "claude-desktop",
    "codex",
    "cursor",
    "opencode",
    "antigravity",
)


def server_command(installed: bool) -> tuple[str, list[str]]:
    """Return (command, args) for the target install mode."""
    if installed:
        return "uvx", ["--from", f"{GIT_URL}@{GIT_REF}", "data-analyst-mcp"]
    return "uv", ["--directory", str(REPO_ROOT), "run", "data-analyst-mcp"]


def claude_desktop_path() -> Path:
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"
    if system == "Windows":
        appdata = os.environ.get("APPDATA")
        if not appdata:
            raise RuntimeError("APPDATA is not set; cannot locate Claude Desktop config")
        return Path(appdata) / "Claude" / "claude_desktop_config.json"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "Claude" / "claude_desktop_config.json"


def codex_path() -> Path:
    return Path.home() / ".codex" / "config.toml"


def claude_code_path() -> Path:
    return REPO_ROOT / ".mcp.json"


def cursor_path() -> Path:
    return REPO_ROOT / ".cursor" / "mcp.json"


def opencode_path() -> Path:
    return REPO_ROOT / "opencode.json"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def mcp_servers_entry(command: str, args: list[str]) -> dict[str, Any]:
    return {"command": command, "args": args}


def install_mcp_servers_json(
    path: Path, name: str, command: str, args: list[str], *, dry_run: bool
) -> str:
    data = load_json(path)
    servers = data.setdefault("mcpServers", {})
    servers[name] = mcp_servers_entry(command, args)
    if dry_run:
        return f"[dry-run] would write {path}:\n{json.dumps(data, indent=2)}"
    write_json(path, data)
    return f"wrote {path}"


def install_opencode_json(
    path: Path, name: str, command: str, args: list[str], *, dry_run: bool
) -> str:
    data = load_json(path)
    data.setdefault("$schema", "https://opencode.ai/config.json")
    mcp = data.setdefault("mcp", {})
    mcp[name] = {
        "type": "local",
        "command": [command, *args],
        "enabled": True,
    }
    if dry_run:
        return f"[dry-run] would write {path}:\n{json.dumps(data, indent=2)}"
    write_json(path, data)
    return f"wrote {path}"


_TOML_SECTION_RE = re.compile(
    r"^\[mcp_servers\.(?P<name>[A-Za-z0-9_\-]+)\]\s*\n(?:(?!^\[).*\n?)*",
    re.MULTILINE,
)


def _format_toml_args(args: list[str]) -> str:
    inner = ", ".join(json.dumps(a) for a in args)
    return f"[{inner}]"


def _toml_section(name: str, command: str, args: list[str]) -> str:
    lines = [f"[mcp_servers.{name}]", f"command = {json.dumps(command)}"]
    if args:
        lines.append(f"args = {_format_toml_args(args)}")
    # Trailing blank line keeps subsequent sections visually separated.
    return "\n".join(lines) + "\n\n"


def install_codex_toml(
    path: Path, name: str, command: str, args: list[str], *, dry_run: bool
) -> str:
    text = ""
    if path.exists() and path.stat().st_size > 0:
        text = path.read_text(encoding="utf-8")
        # Sanity-check that the existing file is parseable TOML before we touch it.
        tomllib.loads(text)
    new_section = _toml_section(name, command, args)

    def replace(match: re.Match[str]) -> str:
        return new_section if match.group("name") == name else match.group(0)

    if _TOML_SECTION_RE.search(text):
        updated = _TOML_SECTION_RE.sub(replace, text)
        if updated == text:
            # No matching section was overwritten; append fresh.
            updated = (text.rstrip() + "\n\n" if text.strip() else "") + new_section
    else:
        updated = (text.rstrip() + "\n\n" if text.strip() else "") + new_section

    if dry_run:
        return f"[dry-run] would write {path}:\n{updated}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(updated, encoding="utf-8")
    return f"wrote {path}"


def install_antigravity(name: str, command: str, args: list[str]) -> str:
    snippet = {
        "mcpServers": {
            name: mcp_servers_entry(command, args),
        }
    }
    return (
        "Antigravity stores its MCP config in an editor-managed file. "
        "Open Antigravity → Agent panel → MCP Servers → Manage MCP Servers "
        '→ "View raw config" and paste this entry into the `mcpServers` '
        f"object:\n\n{json.dumps(snippet, indent=2)}\n"
    )


def install_for_client(client: str, name: str, *, installed: bool, dry_run: bool) -> str:
    command, args = server_command(installed)
    if client == "claude-code":
        return install_mcp_servers_json(claude_code_path(), name, command, args, dry_run=dry_run)
    if client == "claude-desktop":
        return install_mcp_servers_json(claude_desktop_path(), name, command, args, dry_run=dry_run)
    if client == "codex":
        return install_codex_toml(codex_path(), name, command, args, dry_run=dry_run)
    if client == "cursor":
        return install_mcp_servers_json(cursor_path(), name, command, args, dry_run=dry_run)
    if client == "opencode":
        return install_opencode_json(opencode_path(), name, command, args, dry_run=dry_run)
    if client == "antigravity":
        return install_antigravity(name, command, args)
    raise ValueError(f"unknown client: {client}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="install_mcp.py",
        description="Install data-analyst-mcp into a target MCP client.",
    )
    parser.add_argument(
        "client",
        choices=(*CLIENTS, "all"),
        help="which client (or 'all' for every client above)",
    )
    parser.add_argument(
        "--installed",
        action="store_true",
        help="wire `uvx --from git+...@%s` instead of the local checkout" % GIT_REF,
    )
    parser.add_argument(
        "--name",
        default=DEFAULT_NAME,
        help="server name in the client's config (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the planned write without modifying any file",
    )
    args = parser.parse_args(argv)

    targets = CLIENTS if args.client == "all" else (args.client,)
    for client in targets:
        print(f"== {client} ==")
        try:
            print(
                install_for_client(
                    client, args.name, installed=args.installed, dry_run=args.dry_run
                )
            )
        except Exception as exc:  # noqa: BLE001 — surface any client-specific failure
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
