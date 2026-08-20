/**
 * Welcome Screen Component
 *
 * Generic, model-agnostic welcome screen:
 * - Minimal "knowledge agent" wordmark
 * - Centered input card
 * - 4 capability buttons (search / browse / relations / deep reasoning)
 *   instead of hard-coded sample questions (different vaults differ)
 */

"use client"

import React from "react"
import { Brain, Search, FileText, GitFork, Sparkles, Send, Square, Paperclip, Mic } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { FilePreviewGrid } from "./file-preview-grid"
import { VoiceInputButton } from "./voice-input-button"
import type { ImageAttachment } from "@/lib/types"
import type { AgentConfig } from "@/components/layout/agent-settings"
import { MAX_INPUT_CHARS } from "@/lib/constants/features"
import {
  getAllowedModels,
  getModelDisplayName,
  type ModelOption,
} from "@/lib/config/deployment-config"

interface WelcomeScreenProps {
  input: string
  onInputChange: (value: string) => void
  onBeforeInput: (e: React.FormEvent<HTMLTextAreaElement>) => void
  onKeyDown: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void
  isLoading: boolean
  isStopping: boolean
  onStop: () => void
  onSend: () => void
  userId?: string | null

  // File upload
  attachedFiles: ImageAttachment[]
  uploadError: string | null
  inputError: string | null
  isDragging: boolean
  onDragOver: (e: React.DragEvent) => void
  onDragLeave: (e: React.DragEvent) => void
  onDrop: (e: React.DragEvent) => void
  onPaste: (e: React.ClipboardEvent<HTMLTextAreaElement>) => void
  onRemoveFile: (fileId: string) => void
  onFileButtonClick: (e: React.MouseEvent) => void
  fileInputRef: React.RefObject<HTMLInputElement>
  onFileSelect: (e: React.ChangeEvent<HTMLInputElement>) => void
  textareaRef?: React.RefObject<HTMLTextAreaElement>

  // Voice input
  isVoiceListening?: boolean
  isVoiceSupported?: boolean
  onVoiceToggle?: () => void
  voiceError?: string | null

  // Agent configuration
  agentConfig?: AgentConfig
  onAgentConfigChange?: (config: AgentConfig) => void
}

// Capability entries. Kept generic on purpose - we only seed the input
// with a prompt prefix, never with concrete questions (which would assume
// the user's vault content).
const CAPABILITIES = [
  {
    icon: Search,
    label: "检索笔记",
    prompt: "在我的笔记里搜索：",
  },
  {
    icon: FileText,
    label: "浏览笔记库",
    prompt: "打开我的笔记库，查看：",
  },
  {
    icon: GitFork,
    label: "查找关联",
    prompt: "找出和以下主题相关的笔记：",
  },
  {
    icon: Sparkles,
    label: "深度推理",
    prompt: "综合分析（基于我的笔记）：",
  },
] as const

export function WelcomeScreen({
  input,
  onInputChange,
  onBeforeInput,
  onKeyDown,
  isLoading,
  isStopping,
  onStop,
  onSend,
  userId,
  attachedFiles,
  uploadError,
  inputError,
  isDragging,
  onDragOver,
  onDragLeave,
  onDrop,
  onPaste,
  onRemoveFile,
  onFileButtonClick,
  fileInputRef,
  onFileSelect,
  textareaRef,
  isVoiceListening,
  isVoiceSupported,
  onVoiceToggle,
  voiceError,
  agentConfig,
  onAgentConfigChange,
}: WelcomeScreenProps) {
  const allowedModels = getAllowedModels()

  const handleModelChange = (model: string) => {
    if (agentConfig && onAgentConfigChange) {
      onAgentConfigChange({ ...agentConfig, model })
    }
  }

  return (
    <div
      className="absolute inset-0 flex items-center justify-center px-4 overflow-y-auto"
      style={{
        background:
          "radial-gradient(ellipse 60% 50% at 50% 30%, rgba(59,130,246,0.06), transparent 70%), linear-gradient(180deg, #ffffff 0%, #fafbfc 100%)",
      }}
    >
      <div className="w-full max-w-2xl py-12">
        {/* Wordmark - MyKAG */}
        <div className="flex flex-col items-center mb-10 select-none">
          <div className="flex items-center gap-3 text-slate-800">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 shadow-sm">
              <Brain className="h-5 w-5 text-white" />
            </div>
            <span className="text-[28px] font-bold tracking-tight">MyKAG</span>
          </div>
          <span className="mt-1.5 text-xs tracking-[0.2em] text-slate-400 uppercase">
            My Knowledge Agent
          </span>
        </div>

        {/* Input card */}
        <div className="relative group">
          <div
            className={`relative bg-white border rounded-2xl shadow-[0_8px_30px_rgba(0,0,0,0.06)] transition-all duration-200 ${
              isDragging
                ? "border-blue-400 ring-2 ring-blue-100"
                : "border-slate-200 group-focus-within:border-slate-300 group-focus-within:shadow-[0_8px_40px_rgba(0,0,0,0.09)]"
            }`}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
          >
            {isDragging && (
              <div className="absolute inset-0 bg-blue-50/90 rounded-2xl flex items-center justify-center z-20 pointer-events-none">
                <div className="text-blue-600 font-medium text-sm">Drop files to attach</div>
              </div>
            )}

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*,.py,.js,.ts,.tsx,.jsx,.java,.cpp,.c,.h,.cs,.go,.rs,.rb,.php,.sh,.bash,.yaml,.yml,.json,.xml,.html,.css,.md,.txt,.log,.sql,.graphql,.r,.swift,.kt,.scala,.har"
              multiple
              onChange={onFileSelect}
              className="hidden"
            />

            <FilePreviewGrid files={attachedFiles} onRemove={onRemoveFile} />

            <div className="px-4 pt-3 pb-2">
              <Textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => onInputChange(e.target.value)}
                onBeforeInput={onBeforeInput}
                onKeyDown={onKeyDown}
                onPaste={onPaste}
                maxLength={MAX_INPUT_CHARS}
                placeholder={userId ? "有问题尽管问…" : "Initializing…"}
                className="min-h-[28px] max-h-[240px] resize-none bg-transparent border-0 w-full px-0 py-1 text-[15px] leading-7 text-slate-800 placeholder:text-slate-400 focus-visible:ring-0 focus-visible:ring-offset-0 break-words"
                disabled={isLoading || !userId}
                rows={1}
              />
            </div>

            {(uploadError || voiceError) && (
              <div className="px-4 pb-2 text-xs text-destructive">
                {uploadError || voiceError}
              </div>
            )}

            <div className="flex items-center gap-2 px-3 py-2.5 border-t border-slate-100">
              <div className="flex items-center gap-1.5 flex-1 min-w-0">
                {!isLoading && (
                  <Button
                    onClick={onFileButtonClick}
                    variant="ghost"
                    size="icon"
                    disabled={!userId}
                    className="h-8 w-8 rounded-lg text-slate-500 hover:text-slate-700 hover:bg-slate-100"
                    type="button"
                    title="Attach files (images, code, logs)"
                  >
                    <Paperclip className="w-4 h-4" />
                  </Button>
                )}

                {isVoiceSupported && onVoiceToggle && (
                  <VoiceInputButton
                    isListening={isVoiceListening ?? false}
                    disabled={!userId}
                    onClick={onVoiceToggle}
                    size="sm"
                  />
                )}

                {agentConfig && onAgentConfigChange && (
                  <Select value={agentConfig.model} onValueChange={handleModelChange}>
                    <SelectTrigger className="h-8 text-xs border-0 bg-slate-100/80 hover:bg-slate-100 px-2.5 gap-1 w-auto rounded-lg text-slate-600">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {allowedModels.map((model) => (
                        <SelectItem key={model} value={model}>
                          {getModelDisplayName(model as ModelOption)}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}
              </div>

              {isLoading ? (
                <Button
                  onClick={onStop}
                  variant="ghost"
                  size="sm"
                  disabled={isStopping}
                  className={`h-8 px-4 rounded-full gap-1.5 bg-slate-100 text-slate-700 hover:bg-slate-200 ${isStopping ? "opacity-60 cursor-not-allowed" : ""}`}
                  type="button"
                  title={isStopping ? "正在停止..." : "暂停生成"}
                >
                  <Square className="w-3 h-3 fill-current" />
                  <span className="text-xs font-medium">
                    {isStopping ? "停止中…" : "暂停"}
                  </span>
                </Button>
              ) : (
                <Button
                  onClick={onSend}
                  size="sm"
                  disabled={!input.trim() && attachedFiles.length === 0}
                  className="h-8 w-8 p-0 rounded-full bg-slate-900 text-white hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed"
                  type="button"
                  title="发送"
                >
                  <Send className="w-3.5 h-3.5" />
                </Button>
              )}
            </div>
          </div>

          {inputError && (
            <div className="mt-2 px-2 text-xs text-destructive">
              {inputError}
            </div>
          )}
        </div>

        {/* Capability buttons (generic, no hard-coded vault questions) */}
        <div className="grid grid-cols-4 gap-2 mt-8">
          {CAPABILITIES.map(({ icon: Icon, label, prompt }) => (
            <button
              key={label}
              type="button"
              onClick={() => onInputChange(prompt)}
              className="flex flex-col items-center gap-2 p-3 rounded-2xl hover:bg-white/70 transition-all duration-200 group"
            >
              <div className="h-12 w-12 rounded-full bg-white border border-slate-200/80 shadow-sm group-hover:border-blue-300 group-hover:shadow-md group-hover:-translate-y-0.5 flex items-center justify-center transition-all duration-200">
                <Icon className="w-5 h-5 text-slate-500 group-hover:text-blue-600 transition-colors" />
              </div>
              <span className="text-xs text-slate-500 group-hover:text-slate-800 transition-colors">
                {label}
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
