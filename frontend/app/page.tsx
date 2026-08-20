"use client"

import { Suspense, useState, useEffect, useRef } from "react"
import { useQueryState } from "nuqs"
import { Sidebar } from "@/components/layout/sidebar"
import { Header } from "@/components/layout/header"
import { ChatInterface } from "@/components/chat/chat-interface"
import { KeyboardShortcutsDialog } from "@/components/layout/keyboard-shortcuts-dialog"
import { useThreads, type ClientProfile } from "@/lib/hooks/threads"
import { useUserId, useClientProfile } from "@/lib/hooks/auth"
import { resolveClientProfile } from "@/lib/config/client-config"
import type { AgentConfig } from "@/components/layout/agent-settings"
import { generateQuickTitle, generateThreadTitle } from "@/lib/utils/string"
import {
  getAllowedModels,
  getAllowedAgents,
  getDefaultModel,
  getDefaultAgent,
  CONFIG_STORAGE,
  type ModelOption,
  type AgentType,
} from "@/lib/config/deployment-config"
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts"

function DashboardContent() {
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false)
  const [showToolCalls, setShowToolCalls] = useState(false)
  const [showShortcutsDialog, setShowShortcutsDialog] = useState(false)
  const [showSettingsDialog, setShowSettingsDialog] = useState(false)
  const [forceShowTooltip, setForceShowTooltip] = useState(0)

  // Track newly created threads that haven't been initialized in backend yet
  const [newThreads, setNewThreads] = useState<Set<string>>(new Set())

  // Use URL query param for thread ID (shareable, bookmarkable)
  const [threadId, setThreadId] = useQueryState("threadId")

  // Support ?q=... for auto-sending a prompt on page load
  const [initialPrompt, setInitialPrompt] = useQueryState("q")

  const hasInitializedThreadRef = useRef(false)
  const hasHandledInitialPromptRef = useRef(false)

  // Get browser-specific user ID
  const userId = useUserId()

  // Load agent config from localStorage on mount
  const [agentConfig, setAgentConfig] = useState<AgentConfig>(() => {
    if (typeof window !== 'undefined') {
      // Check config version - reset if outdated
      const savedVersion = localStorage.getItem(CONFIG_STORAGE.versionKey)
      if (savedVersion !== CONFIG_STORAGE.version) {
        // Version mismatch - clear old config and set new version
        localStorage.removeItem(CONFIG_STORAGE.key)
        localStorage.setItem(CONFIG_STORAGE.versionKey, CONFIG_STORAGE.version)
        console.log(`Config version updated to ${CONFIG_STORAGE.version}, resetting to defaults`)
      } else {
        const saved = localStorage.getItem(CONFIG_STORAGE.key)
        if (saved) {
          try {
            return JSON.parse(saved)
          } catch (e) {
            console.error('Failed to parse saved agent config:', e)
          }
        }
      }
    }
    // Default config
    return {
      model: getDefaultModel(),
      recursionLimit: 100,
      agentType: getDefaultAgent(),
    }
  })

  // Save agent config to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem(CONFIG_STORAGE.key, JSON.stringify(agentConfig))
  }, [agentConfig])

  // Load threads from LangGraph backend
  const {
    threads,
    isLoading: threadsLoading,
    getThreadById,
    createThread,
    updateThreadMetadata,
    deleteThread,
    addOptimisticThread,
  } = useThreads(userId || undefined)

  const { clientProfile } = useClientProfile()

  // Create a new thread
  const handleNewChat = () => {
    const newThreadId = crypto.randomUUID()

    // Mark this thread as new (doesn't exist in backend yet)
    setNewThreads(prev => new Set(prev).add(newThreadId))

    // Immediately add "Untitled" thread to sidebar
    if (userId) {
      addOptimisticThread({
        thread_id: newThreadId,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        metadata: {
          user_id: userId,
          title: "Untitled",
          lastMessage: "",
          client: resolveClientProfile(clientProfile),
        },
      })
    }

    setThreadId(newThreadId)
  }

  // Switch to an existing thread
  const handleSelectThread = (selectedThreadId: string) => {
    setThreadId(selectedThreadId)
  }

  // Delete a thread
  const handleDeleteThread = (threadIdToDelete: string) => {
    deleteThread(threadIdToDelete, () => {
      // If deleting current thread, create a new one
      if (threadIdToDelete === threadId) {
        const newThreadId = crypto.randomUUID()
        setThreadId(newThreadId)
      }
    })
  }

  // Handle when thread is not found (404) or access denied (403)
  const handleThreadNotFound = () => {
    console.log('Thread not accessible - creating new thread')

    // Always create a new thread when current thread is not accessible
    const newThreadId = crypto.randomUUID()

    // Mark this thread as new (doesn't exist in backend yet)
    setNewThreads(prev => new Set(prev).add(newThreadId))

    // Add to sidebar optimistically
    if (userId) {
      addOptimisticThread({
        thread_id: newThreadId,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        metadata: {
          user_id: userId,
          title: "Untitled",
          lastMessage: "",
          client: resolveClientProfile(clientProfile),
        },
      })
    }

    setThreadId(newThreadId)
  }

  // Update thread metadata when messages are sent
  const handleThreadUpdate = async (
    threadId: string,
    title: string,
    lastMessage: string,
    client?: ClientProfile,
    messageCount?: number, // Track how many messages are in the thread
  ) => {
    if (!userId) return

    // Clear the new thread flag once the thread has been initialized (first message sent)
    if (newThreads.has(threadId)) {
      setNewThreads(prev => {
        const updated = new Set(prev)
        updated.delete(threadId)
        return updated
      })
    }

    const resolvedClient = resolveClientProfile(client ?? clientProfile)

    // Check if this thread already exists
    const existingThread = threads.find(t => t.thread_id === threadId)
    const isUntitledThread = existingThread?.metadata?.title === "Untitled"
    const shouldGenerateAITitle = !existingThread || // First message (thread doesn't exist)
                                  isUntitledThread || // First real message (was "Untitled")
                                  (messageCount && messageCount > 1 && messageCount % 5 === 0) // Every 5 messages after

    if (!existingThread || isUntitledThread) {
      // First message: Keep "Untitled" while AI title generates, then replace directly

      if (!existingThread) {
        // Thread doesn't exist at all - add it with "Untitled"
        addOptimisticThread({
          thread_id: threadId,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          metadata: {
            user_id: userId,
            title: "Untitled",
            lastMessage,
            client: resolvedClient,
          },
        })
      }

      // Update last message immediately (keep "Untitled" for now)
      await updateThreadMetadata(threadId, {
        user_id: userId,
        lastMessage,
        client: resolvedClient,
      })

      // Generate AI title in background - goes straight from "Untitled" to AI title
      generateThreadTitle({
        userMessage: title,
        assistantResponse: lastMessage,
      }).then((aiTitle) => {
        if (aiTitle.length > 0) {
          console.log('Setting AI title:', aiTitle)
          updateThreadMetadata(threadId, {
            user_id: userId,
            title: aiTitle,
            lastMessage,
            client: resolvedClient,
          })
        }
      }).catch((error) => {
        console.error('Failed to generate AI title:', error)
        // Fallback to quick title if AI fails
        const quickTitle = generateQuickTitle(title)
        updateThreadMetadata(threadId, {
          user_id: userId,
          title: quickTitle,
          lastMessage,
          client: resolvedClient,
        })
      })
    } else if (shouldGenerateAITitle && messageCount) {
      // Every 5 messages: Regenerate AI title based on conversation
      console.log(`Regenerating AI title at message ${messageCount}`)

      // Update last message immediately
      await updateThreadMetadata(threadId, {
        user_id: userId,
        lastMessage,
        client: resolvedClient,
      })

      // Generate new AI title in background
      generateThreadTitle({
        userMessage: title,
        assistantResponse: lastMessage,
      }).then((aiTitle) => {
        if (aiTitle.length > 0) {
          console.log('Updated title at message', messageCount, '→', aiTitle)
          updateThreadMetadata(threadId, {
            user_id: userId,
            title: aiTitle,
            lastMessage,
            client: resolvedClient,
          })
        }
      }).catch((error) => {
        console.error('Failed to regenerate AI title:', error)
      })
    } else {
      // Regular update: Just update last message, keep existing title
      await updateThreadMetadata(threadId, {
        user_id: userId,
        lastMessage,
        client: resolvedClient,
      })
    }
  }

  // If no threadId in URL, create one.
  // If ?q= is present, create exactly one fresh thread and mark the prompt as handled.
  // If the current threadId is stale or inaccessible, replace it once with a new thread.
  useEffect(() => {
    let cancelled = false

    const bootstrapThread = async () => {
      // 1) No threadId at all: create a fresh one on the backend.
      if (!threadId && !hasInitializedThreadRef.current) {
        if (!userId) return
        hasInitializedThreadRef.current = true
        const newThreadId = crypto.randomUUID()
        await createThread(newThreadId, { user_id: userId })
        if (cancelled) return
        setNewThreads(prev => new Set(prev).add(newThreadId))
        setThreadId(newThreadId)
        return
      }

      // 2) Got a threadId: verify it exists; if not, create a new one.
      if (threadId) {
        hasInitializedThreadRef.current = true

        // 本地已知的新对话(由 handleNewChat 刚生成的 UUID)跳过 GET:
        // 服务端尚未注册此 id,首次发消息会由 runs.stream(ifNotExists: "create") 自动建。
        // 这样 "点新对话" 不会再触发 404 红色,也不会在 dev server 重启后一上来就刷红屏。
        if (userId && !newThreads.has(threadId)) {
          const existingThread = await getThreadById(threadId)
          if (cancelled) return
          if (!existingThread) {
            const replacementThreadId = crypto.randomUUID()
            await createThread(replacementThreadId, { user_id: userId })
            if (cancelled) return
            setNewThreads(prev => new Set(prev).add(replacementThreadId))
            setThreadId(replacementThreadId)
            return
          }
        }
      }

      // 3) Handle ?q= initial prompt (unchanged behavior).
      const trimmedPrompt = initialPrompt?.trim()
      if (trimmedPrompt && !hasHandledInitialPromptRef.current) {
        hasHandledInitialPromptRef.current = true
        if (!threadId) {
          const newThreadId = crypto.randomUUID()
          setNewThreads(prev => new Set(prev).add(newThreadId))
          setThreadId(newThreadId)
        }
      }
    }

    bootstrapThread()

    return () => {
      cancelled = true
    }
  }, [threadId, setThreadId, initialPrompt, userId, getThreadById, createThread])

  // Cycle to next model
  const handleCycleModel = () => {
    const models = getAllowedModels()
    const currentIndex = models.indexOf(agentConfig.model as ModelOption)
    const nextIndex = (currentIndex + 1) % models.length
    const nextModel = models[nextIndex]
    setAgentConfig({ ...agentConfig, model: nextModel })

    // Trigger the existing tooltip to show
    setForceShowTooltip(prev => prev + 1)
  }

  // Cycle to next agent
  const handleCycleAgent = () => {
    const agents = getAllowedAgents()
    const currentIndex = agents.indexOf(agentConfig.agentType as AgentType)
    const nextIndex = (currentIndex + 1) % agents.length
    const nextAgent = agents[nextIndex]
    setAgentConfig({ ...agentConfig, agentType: nextAgent })

    // Trigger the existing tooltip to show
    setForceShowTooltip(prev => prev + 1)
  }

  // Keyboard shortcuts
  useKeyboardShortcuts([
    {
      shortcut: {
        key: '/',
        metaKey: true,
        description: 'Toggle keyboard shortcuts',
        category: 'Navigation',
      },
      handler: () => setShowShortcutsDialog(!showShortcutsDialog),
    },
    {
      shortcut: {
        key: 'b',
        metaKey: true,
        description: 'Toggle sidebar',
        category: 'Navigation',
      },
      handler: () => setIsSidebarCollapsed(!isSidebarCollapsed),
    },
    {
      shortcut: {
        key: 'i',
        metaKey: true,
        description: 'Create new chat',
        category: 'Navigation',
      },
      handler: handleNewChat,
    },
    {
      shortcut: {
        key: 's',
        metaKey: true,
        description: 'Toggle settings',
        category: 'Navigation',
      },
      handler: () => setShowSettingsDialog(!showSettingsDialog),
    },
    {
      shortcut: {
        key: 'j',
        metaKey: true,
        description: 'Switch model',
        category: 'Model & Agent',
      },
      handler: handleCycleModel,
    },
    {
      shortcut: {
        key: 'k',
        metaKey: true,
        description: 'Switch agent',
        category: 'Model & Agent',
      },
      handler: handleCycleAgent,
    },
  ])

  return (
    <>
      <KeyboardShortcutsDialog
        open={showShortcutsDialog}
        onOpenChange={setShowShortcutsDialog}
      />
      <div className="flex h-screen bg-[linear-gradient(180deg,#f8fbf8_0%,#eef7f0_100%)] text-foreground">
        <Sidebar
          isCollapsed={isSidebarCollapsed}
          onToggle={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
          threads={threads}
          currentThreadId={threadId || ''}
          onSelectThread={handleSelectThread}
          onDeleteThread={handleDeleteThread}
          isLoading={threadsLoading}
        />
        <div className="flex-1 flex flex-col overflow-hidden relative">
          <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(95,181,138,0.12),transparent_32%),radial-gradient(circle_at_bottom_left,rgba(121,200,154,0.08),transparent_30%)]" />
          <div className="relative z-10 flex-1 flex flex-col overflow-hidden">
            <Header
              showToolCalls={showToolCalls}
              onToggleToolCalls={() => setShowToolCalls(!showToolCalls)}
              onNewChat={handleNewChat}
              agentConfig={agentConfig}
              onAgentConfigChange={setAgentConfig}
              onShowShortcuts={() => setShowShortcutsDialog(true)}
              forceShowTooltip={forceShowTooltip}
              showSettingsDialog={showSettingsDialog}
              onSettingsDialogChange={setShowSettingsDialog}
            />
            {threadId ? (
              <ChatInterface
                showToolCalls={showToolCalls}
                threadId={threadId}
                onThreadUpdate={handleThreadUpdate}
                onThreadNotFound={handleThreadNotFound}
                agentConfig={agentConfig}
                onAgentConfigChange={setAgentConfig}
                isNewThread={newThreads.has(threadId)}
                initialMessage={initialPrompt}
                autoSend={!!initialPrompt}
                onInitialMessageSent={() => setInitialPrompt(null)}
              />
            ) : (
              <div className="flex-1 flex items-center justify-center px-4 py-8">
                <div className="rounded-2xl border border-border/60 bg-white/80 px-5 py-4 shadow-sm text-center">
                  <div className="mx-auto mb-3 h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                  <p className="text-sm font-medium text-foreground">正在初始化会话...</p>
                  <p className="mt-1 text-xs text-muted-foreground">请稍候</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  )
}

export default function DashboardPage() {
  return (
    <Suspense fallback={
      <div className="flex h-screen items-center justify-center bg-[linear-gradient(180deg,#f8fbf8_0%,#eef7f0_100%)]">
        <div className="text-center rounded-2xl border border-border/60 bg-white/80 px-6 py-5 shadow-lg backdrop-blur-sm">
          <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-3" />
          <p className="text-sm text-muted-foreground">Loading...</p>
        </div>
      </div>
    }>
      <DashboardContent />
    </Suspense>
  )
}
