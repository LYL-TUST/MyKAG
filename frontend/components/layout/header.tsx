"use client"

import { Brain } from "lucide-react"
import { AgentSettings, type AgentConfig } from "./agent-settings"

interface HeaderProps {
  showToolCalls?: boolean
  onToggleToolCalls?: () => void
  onNewChat?: () => void
  agentConfig?: AgentConfig
  onAgentConfigChange?: (config: AgentConfig) => void
  onShowShortcuts?: () => void
  forceShowTooltip?: number
  showSettingsDialog?: boolean
  onSettingsDialogChange?: (open: boolean) => void
}

export function Header({ showToolCalls = false, onToggleToolCalls, onNewChat, agentConfig, onAgentConfigChange, onShowShortcuts, forceShowTooltip, showSettingsDialog, onSettingsDialogChange }: HeaderProps) {
  return (
    <header className="border-b border-blue-100/80 bg-gradient-to-r from-white via-blue-50/60 to-indigo-50/50 backdrop-blur-xl h-16 flex items-center shadow-[0_1px_0_rgba(59,130,246,0.07)]">
      <div className="flex items-center justify-between w-full px-4 sm:px-6">
        <div className="flex items-center">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-500 via-indigo-500 to-purple-500 shadow-[0_10px_25px_rgba(59,130,246,0.18)] ring-1 ring-white/70">
              <div className="flex h-7 w-7 items-center justify-center rounded-xl bg-white/95 text-blue-600 shadow-sm">
                <Brain className="h-4 w-4" />
              </div>
            </div>
            <div className="flex flex-col gap-0.5">
              <div className="flex items-center gap-2">
                <span className="text-[15px] font-semibold tracking-tight text-slate-800">个人知识管理助手</span>
                <span className="hidden md:inline-flex items-center rounded-full border border-blue-200/80 bg-blue-50 px-2 py-0.5 text-[10px] font-medium text-blue-700">
                  让笔记会说话
                </span>
              </div>
              <span className="hidden lg:inline text-xs text-slate-500">
                基于 Obsidian 知识图谱的智能语义检索与跨文档推理
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {agentConfig && onAgentConfigChange && (
            <AgentSettings
              config={agentConfig}
              onConfigChange={onAgentConfigChange}
              onShowShortcuts={onShowShortcuts}
              forceShowTooltip={forceShowTooltip}
              open={showSettingsDialog}
              onOpenChange={onSettingsDialogChange}
            />
          )}
          <button
            onClick={onNewChat}
            className="group inline-flex items-center gap-2 px-3 sm:px-4 py-2 bg-gradient-to-r from-primary/12 to-primary/6 hover:from-primary/18 hover:to-primary/10 border border-primary/15 hover:border-primary/30 rounded-full text-sm font-medium text-foreground/80 hover:text-foreground transition-all duration-200 hover:-translate-y-0.5 hover:shadow-md shadow-sm"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="text-primary group-hover:rotate-12 transition-transform duration-200"
            >
              <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/>
            </svg>
            <span className="hidden sm:inline">新对话</span>
          </button>
        </div>
      </div>
    </header>
  )
}
