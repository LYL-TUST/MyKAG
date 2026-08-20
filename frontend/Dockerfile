# ---------------------------------------------------------------------------
# Personal Knowledge Agent - frontend (Next.js 16)
# Multi-stage build.
#
# IMPORTANT: NEXT_PUBLIC_LANGGRAPH_API_URL is inlined into the client bundle at
# BUILD time (it runs in the browser, not the container), so it must point at
# the host-published backend URL (http://localhost:3001) - NOT backend:3001.
# ---------------------------------------------------------------------------
FROM node:22-slim AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:22-slim AS builder
WORKDIR /app
ENV NEXT_PUBLIC_LANGGRAPH_API_URL=http://localhost:3001 \
    NODE_ENV=production
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM node:22-slim AS runner
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_PUBLIC_LANGGRAPH_API_URL=http://localhost:3001
# Bring the full built app (includes .next, node_modules, config, source).
COPY --from=builder /app /app
EXPOSE 3000
CMD ["npm", "run", "start"]
