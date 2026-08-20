"""诊断脚本:用独立 QDRANT_PATH 重建索引,跑几次搜索,确认 HTML-CSS 笔记能否被检索到。

用法:.venv/Scripts/python.exe scripts/diag_vault_recall.py
"""
import os
import shutil

# 用完全独立的 Qdrant 目录,避免和 dev server (qdrant_data_v2) 抢文件锁
DIAG_DIR = "./qdrant_diag"
VAULT_PATH = r"E:\简历资料md文档\面试八股文"

if os.path.isdir(DIAG_DIR):
    shutil.rmtree(DIAG_DIR)
os.makedirs(DIAG_DIR, exist_ok=True)

os.environ["QDRANT_PATH"] = DIAG_DIR
os.environ["OBSIDIAN_VAULT_PATH"] = VAULT_PATH

# 项目根
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 手动 load_dotenv + 把所有 key 设进 os.environ
# (vault_tools 的 import 链不经过 src.agent.config,不会自动加载)
import dotenv
dotenv.load_dotenv()

from src.tools.vault_tools import _init_vault, search_vault

print(f"[diag] vault={VAULT_PATH}")
print(f"[diag] qdrant={DIAG_DIR}")

# 强制重建(隔离目录里本来也没有索引)
_init_vault(vault_path=VAULT_PATH, force_rebuild=True)

queries = ["html5的新特性", "html5 新特性", "HTML5", "HTML-CSS", "新特性"]

for q in queries:
    print(f"\n{'='*60}\n[query] {q!r}\n{'='*60}")
    result = search_vault.invoke({"query": q, "top_k": 3, "expand_wikilinks": False})
    print(result[:1500])