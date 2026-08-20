/**
 * Knowledge graph hook.
 *
 * Fetches the full vault [[wikilink]] graph from /graph for visualization.
 */

import { useCallback, useEffect, useState } from "react"
import { LANGGRAPH_API_URL } from "@/lib/constants/api"

export interface GraphNode {
  id: string
  label: string
  path: string
  tags: string[]
  created_at: string | null
}

export interface GraphEdge {
  source: string
  target: string
}

export interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
  total_nodes: number
  total_edges: number
}

export function useGraph() {
  const [data, setData] = useState<GraphData | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    try {
      const res = await fetch(`${LANGGRAPH_API_URL}/graph`)
      if (!res.ok) {
        throw new Error(`Request failed: ${res.status}`)
      }
      setData(await res.json())
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load graph")
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  // Re-fetch the [[wikilink]] graph when the vault is hot-switched
  // (POST /config dispatches a "vault-changed" CustomEvent).
  useEffect(() => {
    const onVaultChanged = () => {
      void refresh()
    }
    window.addEventListener("vault-changed", onVaultChanged)
    return () => window.removeEventListener("vault-changed", onVaultChanged)
  }, [refresh])

  return { data, isLoading, error, refresh }
}
