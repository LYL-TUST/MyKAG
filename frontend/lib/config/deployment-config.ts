/**
 * Public Personal Knowledge Agent configuration.
 *
 * This frontend is public-only: knowledge_agent, anonymous browser identity,
 * and the public model allowlist.
 */

// =============================================================================
// Config Storage
// =============================================================================

/** Bump version to force reset of saved user configs */
export const CONFIG_STORAGE = {
  key: "agent-config",
  versionKey: "agent-config-version",
  version: "0.6",
} as const

// =============================================================================
// Model Registry
// =============================================================================

interface ModelConfig {
  key: string // Internal config key
  id: string // Provider model id
  name: string // Display name
  provider: "siliconflow" | "openai" | "google" | "baseten"
  description?: string
}

/**
 * All available models - single source of truth
 * Keys are short names, values contain full configuration
 */
export const MODELS = {
  "gpt-5.4-mini": {
    key: "gpt-5.4-mini",
    id: "openai:gpt-5.4-mini",
    name: "GPT-5.4 Mini",
    provider: "openai",
    description: "Strongest mini model for coding, computer use, and subagents",
  },
  "deepseek-v4": {
    key: "deepseek-v4",
    id: "deepseek-ai/DeepSeek-V4-Flash",
    name: "DeepSeek V4",
    provider: "siliconflow",
    description: "DeepSeek V4 via SiliconFlow OpenAI-compatible API",
  },
  "qwen3-8b": {
    key: "qwen3-8b",
    id: "Qwen/Qwen3-8B",
    name: "Qwen3 8B",
    provider: "siliconflow",
    description: "Stable and low-cost model for guardrails classification",
  },
  "gemini-3.1-flash-lite": {
    key: "gemini-3.1-flash-lite",
    id: "google_genai:gemini-3.1-flash-lite-preview",
    name: "Gemini 3.1 Flash Lite",
    provider: "google",
    description: "Fastest, most cost-effective Gemini",
  },
  "glm-5": {
    key: "glm-5",
    id: "baseten:zai-org/GLM-5",
    name: "GLM 5",
    provider: "baseten",
    description: "Z.ai GLM 5 served via Baseten",
  },
} as const satisfies Record<string, ModelConfig>

export type ModelKey = keyof typeof MODELS
export type ModelOption = (typeof MODELS)[ModelKey]["id"]

// =============================================================================
// Agent Registry
// =============================================================================

interface AgentConfig {
  id: string
  name: string
  shortName: string
  description?: string
}

export const AGENTS = {
  router: {
    id: "router_agent",
    name: "Auto Router",
    shortName: "Auto Router",
    description: "按任务复杂度自动路由:简单问答/固定任务/复杂任务",
  },
  knowledge: {
    id: "knowledge_agent",
    name: "Knowledge Agent",
    shortName: "Knowledge Agent",
    description: "Personal knowledge management with Obsidian vault",
  },
} as const satisfies Record<string, AgentConfig>

export type AgentKey = keyof typeof AGENTS
export type AgentType = (typeof AGENTS)[AgentKey]["id"]

interface DeploymentConfig {
  models: ModelKey[]
  agents: AgentKey[]
  defaultModel: ModelKey
  defaultAgent: AgentKey
  requiresAuth: boolean
}

const DEPLOYMENT: DeploymentConfig = {
  models: ["gpt-5.4-mini", "deepseek-v4", "qwen3-8b", "gemini-3.1-flash-lite", "glm-5"],
  agents: ["router", "knowledge"],
  defaultModel: "deepseek-v4",
  defaultAgent: "router",
  requiresAuth: false,
}

// =============================================================================
// Core Functions
// =============================================================================

export function getDeploymentConfig(): DeploymentConfig {
  return DEPLOYMENT
}

// =============================================================================
// Model Functions
// =============================================================================

export function getAllowedModels(): ModelOption[] {
  return getDeploymentConfig().models.map((key) => MODELS[key].id)
}

export function getDefaultModel(): ModelOption {
  return MODELS[getDeploymentConfig().defaultModel].id
}

export function isModelAllowed(modelId: ModelOption): boolean {
  return getAllowedModels().includes(modelId)
}

export function getModelDisplayName(modelId: ModelOption): string {
  const model = Object.values(MODELS).find((m) => m.id === modelId)
  return model?.name ?? modelId
}

export function getModelProvider(modelId: ModelOption): string {
  const model = Object.values(MODELS).find((m) => m.id === modelId)
  return model?.provider ?? "openai"
}

// =============================================================================
// Agent Functions
// =============================================================================

export function getAllowedAgents(): AgentType[] {
  return getDeploymentConfig().agents.map((key) => AGENTS[key].id)
}

export function getDefaultAgent(): AgentType {
  return AGENTS[getDeploymentConfig().defaultAgent].id
}

export function isAgentAllowed(agentId: AgentType): boolean {
  return getAllowedAgents().includes(agentId)
}

export function getAgentDisplayName(agentId: AgentType): string {
  const agent = Object.values(AGENTS).find((a) => a.id === agentId)
  return agent?.name ?? agentId
}

export function getAgentShortDisplayName(agentId: AgentType): string {
  const agent = Object.values(AGENTS).find((a) => a.id === agentId)
  return agent?.shortName ?? agentId
}

// =============================================================================
// Auth Functions
// =============================================================================

export function isAuthRequired(): boolean {
  return getDeploymentConfig().requiresAuth
}
