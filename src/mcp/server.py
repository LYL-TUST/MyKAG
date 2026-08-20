"""
MCP Server for Personal Knowledge Agent.

Exposes vault retrieval and codebase search tools via MCP protocol,
allowing external agents (Claude Desktop, Cursor, etc.) to query
the user's Obsidian vault and project source code.

Start:
    python -m src.mcp.server
    # or: python src/mcp/server.py

Configuration (mcp.json / claude_desktop_config.json):
    {
      "mcpServers": {
        "personal-knowledge-agent": {
          "command": "python",
          "args": ["-m", "src.mcp.server"],
          "env": {
            "OBSIDIAN_VAULT_PATH": "E:/agent-projects/obsidian-vault"
          }
        }
      }
    }
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Ensure the project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ── Eagerly initialize the vault pipeline ──
_VAULT_PATH = os.environ.get("OBSIDIAN_VAULT_PATH", ".")

try:
    from src.tools.vault_tools import _init_vault
    _init_vault(_VAULT_PATH)
    logger.info(f"Vault initialized: {_VAULT_PATH}")
except Exception as e:
    logger.warning(f"Vault initialization failed: {e}")


# ── Create MCP Server ──
import mcp.server as mcp_module
from mcp.server import Server

server: Server = Server(
    name="personal-knowledge-agent",
    version="1.0.0",
)


@server.tool(
    name="search_vault",
    description=(
        "语义搜索 Obsidian vault 中的技术笔记。"
        "使用向量 + BM25 混合检索，自动扩展 [[wikilink]] 关联笔记。"
        "返回最相关的笔记片段、来源文件路径和相关性分数。"
    ),
)
def _search_vault_tool(
    query: str,
    top_k: int = 5,
    expand_wikilinks: bool = True,
) -> str:
    """MCP tool: search vault."""
    from src.tools.vault_tools import search_vault as _sv
    return _sv.invoke({
        "query": query,
        "top_k": top_k,
        "expand_wikilinks": expand_wikilinks,
    })


@server.tool(
    name="search_by_tag",
    description=(
        "按标签筛选 Obsidian vault 中的笔记。"
        "例如搜索所有标记为 #ellie 或 #mcp 的笔记。"
    ),
)
def _search_by_tag_tool(
    tag: str,
    top_k: int = 10,
) -> str:
    """MCP tool: search by tag."""
    from src.tools.vault_tools import search_by_tag as _st
    return _st.invoke({"tag": tag, "top_k": top_k})


@server.tool(
    name="list_tags",
    description=(
        "列出 Obsidian vault 中所有使用过的标签。"
    ),
)
def _list_tags_tool() -> str:
    """MCP tool: list tags."""
    from src.tools.vault_tools import list_tags as _lt
    return _lt.invoke({})


@server.tool(
    name="get_note",
    description=(
        "获取指定笔记的完整内容（含 frontmatter 元数据和 [[wikilink]] 关联图）。"
        "参数 note_name 是笔记文件名（不含 .md 后缀）。"
    ),
)
def _get_note_tool(
    note_name: str,
) -> str:
    """MCP tool: get note."""
    from src.tools.vault_tools import get_note as _gn
    return _gn.invoke({"note_name": note_name})


@server.tool(
    name="get_note_graph",
    description=(
        "获取某个笔记的 [[wikilink]] 知识图谱信息。"
        "显示该笔记链接到哪些笔记（out_links）以及哪些笔记链接回来（in_links）。"
    ),
)
def _get_note_graph_tool(
    note_name: str,
) -> str:
    """MCP tool: get note graph."""
    from src.tools.vault_tools import get_note_graph as _gng
    return _gng.invoke({"note_name": note_name})


@server.tool(
    name="search_codebase",
    description=(
        "搜索项目源码中与查询相关的文件。"
        "支持在 ellie 和 code-review 两个项目中搜索。"
        "返回匹配文件的路径和代码片段。"
        "适用于查找具体实现逻辑、配置文件、函数定义等。"
    ),
)
def _search_codebase_tool(
    query: str,
    project: str = "ellie",
    top_k: int = 5,
) -> str:
    """MCP tool: search codebase."""
    from src.tools.codebase_tools import search_codebase as _sc
    return _sc.invoke({
        "query": query,
        "project": project,
        "top_k": top_k,
    })


@server.tool(
    name="list_codebase_projects",
    description=(
        "列出所有可搜索的项目代码库名称和路径。"
    ),
)
def _list_codebase_projects_tool() -> str:
    """MCP tool: list codebase projects."""
    from src.tools.codebase_tools import list_codebase_projects as _lcp
    return _lcp.invoke({})


# ── Entry Point ──
def main():
    """Start the MCP server via stdio transport."""
    import asyncio
    from mcp.server.stdio import stdio_server

    logger.info("Starting Personal Knowledge Agent MCP Server...")
    logger.info(f"Vault path: {_VAULT_PATH}")
    logger.info(f"7 tools registered: search_vault, search_by_tag, list_tags, "
                f"get_note, get_note_graph, search_codebase, list_codebase_projects")

    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    asyncio.run(run())


if __name__ == "__main__":
    main()
