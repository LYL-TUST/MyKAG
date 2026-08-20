/**
 * API Constants
 *
 * Configuration constants for API endpoints and keys.
 */

/**
 * Get the public LangGraph API URL.
 *
 * NEXT_PUBLIC_LANGGRAPH_API_URL points to the public docs-agent deployment.
 * NEXT_PUBLIC_LANGGRAPH_API_URL_EXTERNAL is supported for compatibility with
 * older frontend deployments.
 */
function getLangGraphApiUrl(): string {
  const url =
    process.env.NEXT_PUBLIC_LANGGRAPH_API_URL ||
    process.env.NEXT_PUBLIC_LANGGRAPH_API_URL_EXTERNAL ||
    (process.env.NODE_ENV === "development" ? "http://127.0.0.1:3001" : "http://127.0.0.1:3001")

  if (!process.env.NEXT_PUBLIC_LANGGRAPH_API_URL && !process.env.NEXT_PUBLIC_LANGGRAPH_API_URL_EXTERNAL) {
    console.warn(
      "[LangGraph] NEXT_PUBLIC_LANGGRAPH_API_URL is not set; falling back to http://127.0.0.1:3001"
    )
  }

  if (process.env.NODE_ENV === "development") {
    console.info("[LangGraph] Public deployment routing to:", url)
  }

  return url
}

export const LANGGRAPH_API_URL = getLangGraphApiUrl()

// LangGraph Server API key (used for LangGraph client, not LangSmith)
// Note: This should be undefined in browser for security
// LangGraph Cloud deployments need auth disabled or custom auth configured
export const LANGSMITH_API_KEY = process.env.LANGSMITH_API_KEY

