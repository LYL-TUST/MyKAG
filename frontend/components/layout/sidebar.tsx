"use client"

import { useState, useMemo, memo, useCallback } from "react"
import { Trash2, PanelLeftClose, PanelLeft, Search, X, MessageSquare, FileText, Settings } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import type { Thread } from "@/lib/hooks/threads"
import { NotesPanel } from "@/components/notes/notes-panel"
import { SettingsDialog } from "@/components/settings/settings-dialog"

// Add custom scrollbar styles - overlay scrollbar that doesn't affect layout
const scrollbarStyles = `
  .custom-scrollbar {
    scrollbar-width: thin;
    scrollbar-color: transparent transparent;
  }
  .custom-scrollbar:hover {
    scrollbar-color: var(--langchain-blue, #7FC8FF) transparent;
  }
  .custom-scrollbar::-webkit-scrollbar {
    width: 6px;
  }
  .custom-scrollbar::-webkit-scrollbar-track {
    background: transparent;
  }
  .custom-scrollbar::-webkit-scrollbar-thumb {
    background: transparent;
    border-radius: 3px;
  }
  .custom-scrollbar:hover::-webkit-scrollbar-thumb {
    background: var(--langchain-blue, #7FC8FF);
  }
  .custom-scrollbar::-webkit-scrollbar-thumb:hover {
    background: var(--langchain-blue, #99D3FF);
  }
`

interface SidebarProps {
  isCollapsed: boolean
  onToggle: () => void
  threads: Thread[]
  currentThreadId: string
  onSelectThread: (threadId: string) => void
  onDeleteThread: (threadId: string) => void
  isLoading?: boolean
}

function getRelativeTime(date: Date): string {
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMins = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMs / 3600000)
  const diffDays = Math.floor(diffMs / 86400000)

  if (diffMins < 1) return "Just now"
  if (diffMins < 60) return `${diffMins} min${diffMins > 1 ? "s" : ""} ago`
  if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? "s" : ""} ago`
  if (diffDays === 1) return "Yesterday"
  if (diffDays < 7) return `${diffDays} days ago`
  if (diffDays < 30) return `${Math.floor(diffDays / 7)} week${Math.floor(diffDays / 7) > 1 ? "s" : ""} ago`
  return `${Math.floor(diffDays / 30)} month${Math.floor(diffDays / 30) > 1 ? "s" : ""} ago`
}

function groupThreads(threads: Thread[]) {
  const now = new Date()
  const today: Thread[] = []
  const yesterday: Thread[] = []
  const last7Days: Thread[] = []
  const older: Thread[] = []

  threads.forEach((thread) => {
    // Use updated_at from LangGraph thread
    const threadDate = new Date(thread.updated_at || thread.created_at)
    const diffMs = now.getTime() - threadDate.getTime()
    const diffHours = diffMs / 3600000
    const diffDays = diffMs / 86400000

    if (diffHours < 24) {
      today.push(thread)
    } else if (diffDays < 2) {
      yesterday.push(thread)
    } else if (diffDays < 7) {
      last7Days.push(thread)
    } else {
      older.push(thread)
    }
  })

  return { today, yesterday, last7Days, older }
}

export const Sidebar = memo(function Sidebar({
  isCollapsed,
  onToggle,
  threads,
  currentThreadId,
  onSelectThread,
  onDeleteThread,
  isLoading = false,
}: SidebarProps) {
  const [searchQuery, setSearchQuery] = useState('')
  const [activeTab, setActiveTab] = useState<'chat' | 'notes'>('chat')
  const [settingsOpen, setSettingsOpen] = useState(false)

  // Filter threads based on search query
  const filteredThreads = useMemo(() => {
    if (!searchQuery.trim()) return threads

    const query = searchQuery.toLowerCase()
    return threads.filter(thread => {
      const title = thread.metadata?.title?.toLowerCase() || ''
      const lastMessage = thread.metadata?.lastMessage?.toLowerCase() || ''
      return title.includes(query) || lastMessage.includes(query)
    })
  }, [threads, searchQuery])

  // Memoize grouped threads to avoid recalculating on every render
  const groupedThreads = useMemo(() => groupThreads(filteredThreads), [filteredThreads])
  const { today, yesterday, last7Days, older } = groupedThreads

  // Memoize event handlers to prevent unnecessary re-renders
  const handleSelectThread = useCallback((threadId: string) => {
    onSelectThread(threadId)
  }, [onSelectThread])

  const handleDeleteThread = useCallback((threadId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    onDeleteThread(threadId)
  }, [onDeleteThread])

  const handleClearSearch = useCallback(() => {
    setSearchQuery('')
  }, [])

  // Memoize renderThreadGroup to prevent recreation on every render
  // IMPORTANT: Must be defined before any conditional returns (Rules of Hooks)
  const renderThreadGroup = useCallback((groupThreads: Thread[], label: string) => {
    if (groupThreads.length === 0) return null

    return (
      <div className="mt-4 px-3 first:mt-0">
        <h3 className="px-3 text-xs font-semibold text-sidebar-accent-foreground uppercase tracking-wider mb-2 shadow-inset-light">{label}</h3>
        <div className="space-y-2">
          {groupThreads.map((thread) => {
            const threadDate = new Date(thread.updated_at || thread.created_at)
            const title = thread.metadata?.title || "New conversation"

            return (
              <div
                key={thread.thread_id}
                className={`group flex items-center gap-3 px-3 py-2.5 text-sm w-full rounded-lg transition-all duration-200 cursor-pointer shadow-depth-xs ${
                  thread.thread_id === currentThreadId
                    ? "bg-[#7FC8FF]/15 text-sidebar-foreground shadow-depth-sm border border-[#7FC8FF]/40"                    : "text-sidebar-foreground"
                }`}
                onClick={() => handleSelectThread(thread.thread_id)}
              >
                <div className="flex-1 min-w-0">
                  <div className="truncate font-medium">{title}</div>
                  <div className="text-xs text-muted-foreground mt-0.5">
                    {getRelativeTime(threadDate)}
                  </div>
                </div>
                <button
                  onClick={(e) => handleDeleteThread(thread.thread_id, e)}
                  className="opacity-0 group-hover:opacity-100 transition-all duration-200 p-1 rounded-md hover:bg-destructive/10"
                >
                  <Trash2 className="w-3.5 h-3.5 text-muted-foreground hover:text-destructive" />
                </button>
              </div>
            )
          })}
        </div>
      </div>
    )
  }, [currentThreadId, handleSelectThread, handleDeleteThread])

  // Early return for collapsed state (after all hooks)
  if (isCollapsed) {
    return (
      <aside className="hidden md:flex w-16 bg-white/90 backdrop-blur-xl border-r border-border/70 flex-col shadow-[1px_0_0_rgba(95,181,138,0.06)]">
        <div className="px-3 py-4 border-b border-border/60 h-16 flex items-center justify-center">
          <Button variant="ghost" size="icon" onClick={onToggle} className="hover:bg-sidebar-primary/10 hover:text-sidebar-primary transition-all duration-200 shadow-sm hover:shadow-md rounded-xl">
            <PanelLeft className="w-5 h-5" />
          </Button>
        </div>
      </aside>
    )
  }

  return (
    <>
      <style>{scrollbarStyles}</style>
      <aside className="hidden md:flex w-56 bg-gradient-to-b from-sidebar via-sidebar-light to-sidebar-lighter border-r border-border/60 flex-col shadow-depth-md">
        <div className="px-3 pt-[13px] pb-2 border-b border-border/60 bg-gradient-to-r from-sidebar-accent/20 via-sidebar-accent/10 to-transparent">
          <div className="flex items-center justify-between gap-2">
            <Button variant="ghost" size="icon" onClick={onToggle} className="hover:bg-sidebar-primary/10 hover:text-sidebar-primary transition-all duration-200 shadow-sm hover:shadow-md rounded-xl shrink-0">
              <PanelLeftClose className="w-5 h-5" />
            </Button>
            <div className="flex items-center gap-1 bg-muted/60 rounded-lg p-0.5">
              <button
                type="button"
                onClick={() => setActiveTab('chat')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all duration-200 ${
                  activeTab === 'chat'
                    ? "bg-background text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                <MessageSquare className="w-3.5 h-3.5" />
                对话
              </button>
              <button
                type="button"
                onClick={() => setActiveTab('notes')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all duration-200 ${
                  activeTab === 'notes'
                    ? "bg-background text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                <FileText className="w-3.5 h-3.5" />
                笔记
              </button>
            </div>
          </div>
        </div>

      {activeTab === 'chat' && (
        <>
      {/* Search Bar */}
      <div className="px-3 py-2 bg-gradient-to-r from-sidebar-accent/5 via-transparent to-transparent">
        <div className="relative group">
          <div className="absolute left-3 top-1/2 transform -translate-y-1/2 z-10">
            <Search className="w-4 h-4 text-muted-foreground/70 group-focus-within:text-primary transition-all duration-200" />
          </div>
          <Input
            type="text"
            placeholder="搜索对话..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10 pr-8 h-10 text-sm bg-background/80 backdrop-blur-sm border-border/40 focus:border-primary/60 focus:bg-background/90 focus:shadow-sm transition-all duration-200 shadow-sm hover:shadow-md hover:bg-background/90 rounded-lg"
          />
          {searchQuery && (
            <button
              type="button"
              onClick={handleClearSearch}
              className="absolute right-3 top-1/2 transform -translate-y-1/2 z-10 text-muted-foreground/60 hover:text-foreground transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-primary/30 rounded-full p-0.5 hover:bg-muted/50"
              aria-label="Clear search"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      <nav className="flex-1 overflow-y-auto py-2 bg-gradient-to-b from-sidebar-accent/5 via-transparent to-sidebar-accent/10 custom-scrollbar">
        {isLoading ? (
          <div className="px-6 py-8 text-center">
            <div className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-3" />
            <p className="text-xs text-muted-foreground">正在加载对话...</p>
          </div>
        ) : searchQuery && filteredThreads.length === 0 ? (
          <div className="px-6 py-8 text-center text-sm text-muted-foreground bg-gradient-to-br from-card/10 via-card/5 to-transparent rounded-lg mx-3 shadow-depth-xs">
            <div className="font-medium mb-1">未找到结果</div>
            <div className="text-xs">请尝试其他搜索词</div>
          </div>
        ) : filteredThreads.length === 0 ? (
          <div className="px-6 py-8 text-center text-sm text-muted-foreground bg-gradient-to-br from-card/10 via-card/5 to-transparent rounded-lg mx-3 shadow-depth-xs">
            <div className="font-medium mb-1">还没有对话</div>
            <div className="text-xs">开始提问后，这里会显示你的对话记录</div>
          </div>
        ) : (
          <>
            {renderThreadGroup(today, "Today")}
            {renderThreadGroup(yesterday, "Yesterday")}
            {renderThreadGroup(last7Days, "Previous 7 Days")}
            {renderThreadGroup(older, "Older")}
          </>
        )}
      </nav>
        </>
      )}

      {activeTab === 'notes' && (
        <div className="flex-1 min-h-0 flex flex-col">
          <NotesPanel />
        </div>
      )}

      <div className="bg-gradient-to-t from-sidebar-accent/10 via-sidebar-accent/5 to-transparent pt-2 pb-3 px-3">
        <button
          type="button"
          onClick={() => setSettingsOpen(true)}
          className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-xs text-muted-foreground hover:bg-sidebar-accent/10 hover:text-sidebar-foreground transition-all"
        >
          <Settings className="w-3.5 h-3.5" />
          设置
        </button>
      </div>

      <SettingsDialog
        open={settingsOpen}
        onOpenChange={setSettingsOpen}
        onVaultChanged={() => {
          // Force notes/graph views to refresh after switching vault
          window.dispatchEvent(new CustomEvent("vault-changed"))
        }}
      />
    </aside>
    </>
  )
})
