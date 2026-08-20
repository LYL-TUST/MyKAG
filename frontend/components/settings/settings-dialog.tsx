"use client"

import { useState } from "react"
import { Settings, Loader2, FolderOpen, CheckCircle2 } from "lucide-react"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { useConfig } from "@/lib/hooks/notes/use-config"

interface SettingsDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onVaultChanged?: () => void
}

export function SettingsDialog({ open, onOpenChange, onVaultChanged }: SettingsDialogProps) {
  const { config, isLoading, error, isSwitching, updateVault } = useConfig()
  const [pathInput, setPathInput] = useState("")
  const [localError, setLocalError] = useState<string | null>(null)
  const [successMsg, setSuccessMsg] = useState<string | null>(null)

  const handleSwitch = async () => {
    const path = pathInput.trim()
    if (!path) {
      setLocalError("请输入 vault 路径")
      return
    }
    setLocalError(null)
    setSuccessMsg(null)
    try {
      const data = await updateVault(path)
      setSuccessMsg(
        `切换成功：${data.graph_stats.nodes} 篇笔记、${data.graph_stats.links} 条关联`
      )
      setPathInput("")
      onVaultChanged?.()
    } catch (e) {
      setLocalError(e instanceof Error ? e.message : "切换失败")
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Settings className="w-4 h-4" />
            设置
          </DialogTitle>
        </DialogHeader>

        {/* Current vault info */}
        <div className="text-sm">
          <div className="font-medium mb-1.5 flex items-center gap-1.5">
            <FolderOpen className="w-3.5 h-3.5 text-muted-foreground" />
            当前知识库
          </div>
          {isLoading ? (
            <div className="text-xs text-muted-foreground">加载中...</div>
          ) : (
            <>
              <div className="text-xs text-muted-foreground break-all bg-muted/50 rounded-md px-3 py-2 font-mono">
                {config?.vault_path || "未配置"}
              </div>
              {config && (
                <div className="text-xs text-muted-foreground mt-1.5">
                  {config.graph_stats.nodes} 篇笔记 · {config.graph_stats.links} 条关联
                  {config.initialized ? " · 已索引" : " · 未索引"}
                </div>
              )}
              <div className="text-xs text-muted-foreground mt-1.5 leading-relaxed">
                知识库范围：你的 Obsidian vault —— 笔记库检索为空时，才会用 LLM 通用知识回答。
              </div>
            </>
          )}
        </div>

        <div className="border-t border-border/60" />

        {/* Switch vault */}
        <div className="text-sm">
          <div className="font-medium mb-1.5">切换知识库</div>
          <p className="text-xs text-muted-foreground mb-2 leading-relaxed">
            输入 Obsidian vault 文件夹的绝对路径（例如 <span className="font-mono">E:/my-vault</span>），切换后自动重建索引，无需重启。
          </p>
          <div className="flex gap-2">
            <Input
              value={pathInput}
              onChange={(e) => setPathInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleSwitch()
              }}
              placeholder="E:/path/to/vault"
              className="flex-1"
            />
            <Button onClick={handleSwitch} disabled={isSwitching} className="shrink-0">
              {isSwitching ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  索引中...
                </>
              ) : (
                "切换"
              )}
            </Button>
          </div>

          {localError && (
            <div className="text-xs text-destructive mt-1.5">{localError}</div>
          )}
          {error && !localError && (
            <div className="text-xs text-destructive mt-1.5">{error}</div>
          )}
          {successMsg && (
            <div className="text-xs text-green-600 mt-1.5 flex items-center gap-1">
              <CheckCircle2 className="w-3.5 h-3.5" />
              {successMsg}
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
