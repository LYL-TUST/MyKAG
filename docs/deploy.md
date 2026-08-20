# 线上部署手册（秋招可讲版）

本项目已容器化（backend + Qdrant + frontend 三服务，`docker-compose.yml`）。
本手册给出两种落地路径：**云服务器自托管**（最稳，后端岗高频考点）与
**Serverless / PaaS 平台**（最快，适合 demo 上线）。

> 前提：Docker / Docker Compose 已安装。`src/rag/indexer.py` 已支持
> `QDRANT_URL`——设了走 **server 模式**（多进程共享、无文件锁），不设则
> 退回 local 模式（开发机单进程）。**线上部署一律用 server 模式**。

---

## 路径 A：云服务器自托管（推荐，最像生产）

适用：阿里云 ECS / 腾讯云 CVM / 华为云 ECS，2 核 4G 起。
目标：公网域名 + HTTPS，三个服务常驻。

### 1. 准备服务器
```bash
# 以 Ubuntu 22.04 为例
sudo apt update && sudo apt install -y docker.io docker-compose-plugin nginx certbot python3-certbot-nginx
sudo usermod -aG docker $USER   # 退出重登生效
```

### 2. 拉代码 + 配环境变量
```bash
git clone <your-repo> personal-knowledge-agent && cd personal-knowledge-agent
cp .env.example .env
# 编辑 .env，至少填：
#   OPENAI_API_KEY=sk-...          # SiliconFlow / 你的 LLM Key
#   OPENAI_BASE_URL=https://api.siliconflow.cn/v1
#   QDRANT_URL=http://qdrant:6333 # 走 compose 内的 Qdrant server
#   OBSIDIAN_VAULT_PATH=/app/vault # 容器内挂载点（见下）
```

### 3. 挂真 vault + 起服务
把你的 Obsidian 笔记目录传到服务器，用 compose 的 `VAULT_HOST_PATH` 挂载
（只读）：
```bash
# 在 docker-compose.yml 里 backend 的 volumes 已写好：
#   ${VAULT_HOST_PATH}:/app/vault:ro
# 默认 VAULT_HOST_PATH=../obsidian-vault，改成你笔记的真实绝对路径即可
export VAULT_HOST_PATH=/root/obsidian-vault
docker compose up -d --build
```
验证：`curl http://localhost:3001/ok` 与 `curl http://localhost:3000` 应正常。

### 4. Nginx 反代 + HTTPS
```nginx
# /etc/nginx/sites-available/pka
server {
    listen 80; server_name pka.your-domain.com;
    location /api/  { proxy_pass http://127.0.0.1:3001/; proxy_set_header Host $host; }
    location /      { proxy_pass http://127.0.0.1:3000; }
}
```
```bash
sudo ln -s /etc/nginx/sites-available/pka /etc/nginx/sites-enabled/
sudo certbot --nginx -d pka.your-domain.com   # 自动签 HTTPS 并跳转 443
```

> 注意：前端 `NEXT_PUBLIC_LANGGRAPH_API_URL` 在 **构建期** 内联。
> 改域名后需重新 `docker compose up -d --build frontend`，或在 `frontend/.env`
> 里先写好生产域名再 build。

### 5. 运维
- 升级：`git pull && docker compose up -d --build`
- 日志：`docker compose logs -f backend`
- 索引持久化：Qdrant 数据在 `qdrant_data` 卷，重装容器不丢
- 重启自起：`docker compose up -d` 已带 `restart: unless-stopped`

---

## 路径 B：Serverless / PaaS（最快上线）

| 平台 | 做法 | 备注 |
|---|---|---|
| **Render** | 连 GitHub，Backend 用 `Dockerfile`（暴露 3001），Frontend 用 `frontend/Dockerfile`（暴露 3000）；加一个 Qdrant 的 "Private Service" | 免费层够 demo；环境变量在后台填 |
| **Fly.io** | `fly launch` 读 `Dockerfile`；Qdrant 用 `fly postgres` 替代或官方 Qdrant 镜像 | 适合海外节点 |
| **Railway** | 连仓库，自动识别 Dockerfile；加 Qdrant 模板 | 最省心 |
| **腾讯云 CloudBase / 阿里云函数计算** | 后端用 Web 服务（容器），前端用静态托管 | 国内访问快；需配 VPC 让 Qdrant 互通 |

通用要点：
- 平台环境变量里填 `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `QDRANT_URL`
  / `OBSIDIAN_VAULT_PATH`，**不要写进镜像**。
- Vault 笔记用平台「Volume / 持久化目录」挂载，或先打包进镜像
  （demo 量小可接受，但更新笔记要重 build）。

---

## 面试怎么讲（话术）

> “CI/CD 我用了 GitHub Actions：push/PR 自动跑 86 个免凭据单测（LLM 调用
> 全 mock，不需要 Key）+ 两个 Docker 镜像构建校验，绿了才允许合入。
> 部署我选了 docker-compose 三服务（Qdrant 独立容器做向量存储，backend
> 走 server 模式避免文件锁），上云用 Nginx 反代 + Let's Encrypt 做 HTTPS。
> 这块我特意把‘凭证不进镜像、Qdrant 数据卷持久化、前端 API 地址构建期内联’
> 三个生产坑处理掉了。”

---

## 已知边界（诚实说明）
- 当前后端用 `langgraph dev`（开发服务器）。**生产建议**换成
  `langgraph-api` 或 `gunicorn` 多 worker；本仓库已预留 `langgraph-api` 依赖。
- 未做自动伸缩 / 监控告警（Prometheus）。demo 阶段非必需，面试可主动提
  “下一步会接”。
