/**
 * LangGraph Client Factory
 *
 * Creates authenticated LangGraph SDK clients for API requests.
 * All clients include Authorization header with user ID for backend auth.
 */

import { Client } from "@langchain/langgraph-sdk"
import { LANGGRAPH_API_URL, LANGSMITH_API_KEY } from "@/lib/constants/api"

/**
 * Create a LangGraph client instance with authentication.
 *
 * @param userId - User ID for Authorization header (required - backend enforces auth)
 * @throws Error if userId is not provided
 *
 * @example
 * ```typescript
 * const client = createLangGraphClient(userId)
 * const threads = await client.threads.search({ metadata: { user_id: userId } })
 * ```
 */
export function createLangGraphClient(userId: string | undefined): Client {
  if (!userId) {
    throw new Error(
      "User ID required for authentication. Ensure user is logged in before making requests."
    )
  }

  const headers: Record<string, string> = {
    Authorization: `Bearer ${userId}`,
  }

  // Optional public app key for deployments that set LANGGRAPH_AUTH_SECRET.
  const authKey = process.env.NEXT_PUBLIC_LANGGRAPH_AUTH_KEY
  if (authKey) {
    headers["X-Auth-Key"] = authKey
  }

  return new Client({
    apiUrl: LANGGRAPH_API_URL,
    apiKey: LANGSMITH_API_KEY,
    defaultHeaders: headers,
  })
}

// Module-level cache: `${apiUrl}::${graphId}` -> assistant_id
// Avoids a search/create round-trip on every message.
const assistantIdCache = new Map<string, string>()

/**
 * Resolve an assistant_id for a given graph_id.
 *
 * langgraph-api v0.12+ requires runs to be started against an *assistant*
 * (assistant_id), not a raw graph_id. Passing a graph_id where the SDK expects
 * an assistant_id produces a 404 and the stream silently yields no messages
 * (the "未生成回答" / "no response generated" symptom).
 *
 * This helper looks up an existing assistant bound to the graph, or creates one
 * if none exists, and returns its assistant_id. Results are cached per graph.
 *
 * @param client - Authenticated LangGraph SDK client.
 * @param graphId - Graph id as registered in langgraph.json (e.g. "router_agent").
 * @returns The assistant_id to use when creating a run.
 */
export async function resolveAssistantId(
  client: Client,
  graphId: string
): Promise<string> {
  const cacheKey = `${LANGGRAPH_API_URL}::${graphId}`
  const cached = assistantIdCache.get(cacheKey)
  if (cached) return cached

  const existing = await client.assistants.search({ graphId, limit: 1 })
  const assistantId =
    existing[0]?.assistant_id ??
    (await client.assistants.create({ graphId })).assistant_id

  assistantIdCache.set(cacheKey, assistantId)
  return assistantId
}
