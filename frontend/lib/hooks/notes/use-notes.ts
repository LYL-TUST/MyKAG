/**
 * Notes browsing hook.
 *
 * Fetches the vault note list, individual note content, and wikilink graph
 * from the backend REST endpoints (/notes).
 */

import { useCallback, useEffect, useState } from "react"
import { LANGGRAPH_API_URL } from "@/lib/constants/api"

export interface NoteSummary {
  name: string
  title: string
  tags: string[]
  wikilink_count: number
  file_path: string
}

export interface NoteDetail extends NoteSummary {
  content: string
  frontmatter: Record<string, unknown>
  wikilinks: string[]
}

export interface NoteGraph {
  node: string
  out_links: string[]
  in_links: string[]
}

export interface NotesState {
  notes: NoteSummary[]
  isLoading: boolean
  error: string | null
  refresh: () => Promise<void>
  getNote: (name: string) => Promise<NoteDetail | null>
  getNoteGraph: (name: string) => Promise<NoteGraph | null>
}

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(`${LANGGRAPH_API_URL}${path}`)
  if (!res.ok) {
    throw new Error(`Request failed: ${res.status} ${res.statusText}`)
  }
  return res.json() as Promise<T>
}

export function useNotes(): NotesState {
  const [notes, setNotes] = useState<NoteSummary[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    try {
      const data = await fetchJson<{ notes: NoteSummary[] }>("/notes")
      setNotes(data.notes ?? [])
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load notes")
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  // Re-fetch when the vault is hot-switched (POST /config) so the notes
  // list reflects the newly active knowledge base without a full reload.
  useEffect(() => {
    const onVaultChanged = () => {
      void refresh()
    }
    window.addEventListener("vault-changed", onVaultChanged)
    return () => window.removeEventListener("vault-changed", onVaultChanged)
  }, [refresh])

  const getNote = useCallback(async (name: string): Promise<NoteDetail | null> => {
    try {
      return await fetchJson<NoteDetail>(`/notes/${encodeURIComponent(name)}`)
    } catch (e) {
      console.error("Failed to fetch note:", e)
      return null
    }
  }, [])

  const getNoteGraph = useCallback(async (name: string): Promise<NoteGraph | null> => {
    try {
      return await fetchJson<NoteGraph>(`/notes/${encodeURIComponent(name)}/graph`)
    } catch (e) {
      console.error("Failed to fetch note graph:", e)
      return null
    }
  }, [])

  return { notes, isLoading, error, refresh, getNote, getNoteGraph }
}
