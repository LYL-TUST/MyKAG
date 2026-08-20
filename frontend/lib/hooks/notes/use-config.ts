/**
 * Configuration hook.
 *
 * Reads the current vault config and allows hot-switching the vault path
 * (POST /config triggers re-indexing without a restart).
 */

import { useCallback, useEffect, useState } from "react"
import { LANGGRAPH_API_URL } from "@/lib/constants/api"

export interface ConfigData {
  vault_path: string
  initialized: boolean
  graph_stats: {
    nodes: number
    links: number
  }
}

export function useConfig() {
  const [config, setConfig] = useState<ConfigData | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [isSwitching, setIsSwitching] = useState(false)

  const refresh = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    try {
      const res = await fetch(`${LANGGRAPH_API_URL}/config`)
      if (!res.ok) throw new Error(`Request failed: ${res.status}`)
      setConfig(await res.json())
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load config")
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  const updateVault = useCallback(async (path: string): Promise<ConfigData> => {
    setIsSwitching(true)
    setError(null)
    try {
      const res = await fetch(`${LANGGRAPH_API_URL}/config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ vault_path: path }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => null)
        throw new Error(body?.detail || `Request failed: ${res.status}`)
      }
      const data = await res.json()
      setConfig(data)
      return data
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Failed to update vault"
      setError(msg)
      throw e
    } finally {
      setIsSwitching(false)
    }
  }, [])

  return { config, isLoading, error, isSwitching, refresh, updateVault }
}
