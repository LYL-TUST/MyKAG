"use client"

import { useMemo, useRef, useState } from "react"
import dynamic from "next/dynamic"
import { Loader2, X, ArrowLeft, GitFork, Download } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useGraph } from "@/lib/hooks/notes/use-graph"

// ForceGraph2D relies on window (canvas), so it must be client-only.
const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false })

// Tech-feel palette: lower-saturation Tailwind 400 series for soft glow on
// the dark background. Hub (moc) gets a brighter red to stand out.
const TAG_COLORS: Record<string, string> = {
  ellie: "#60a5fa",
  "code-review": "#34d399",
  technique: "#a78bfa",
  "job-hunt": "#fbbf24",
  moc: "#f87171",
}

const TAG_LABELS: Record<string, string> = {
  ellie: "ellie 系列",
  "code-review": "Code Review",
  technique: "技术积累",
  "job-hunt": "求职准备",
  moc: "知识总览",
}

const FALLBACK_COLOR = "#94a3b8"
const BG_COLOR = "#0a0e27"

function getNodeColor(tags: string[] | undefined): string {
  if (!tags || tags.length === 0) return FALLBACK_COLOR
  for (const t of tags) {
    if (TAG_COLORS[t]) return TAG_COLORS[t]
  }
  return FALLBACK_COLOR
}

function formatDate(iso: string | null): string {
  if (!iso) return "未知"
  // Strip sub-seconds for compact display.
  return iso.replace("T", " ").slice(0, 16)
}

interface GraphViewProps {
  onClose: () => void
  onOpenNote?: (name: string) => void
}

export function GraphView({ onClose, onOpenNote }: GraphViewProps) {
  const { data, isLoading, error } = useGraph()
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  // Time filter as a fraction [0..1] mapped onto [min..max] mtime.
  // 1 = show everything, 0 = show only the earliest note.
  const [timeFraction, setTimeFraction] = useState(1)

  // Pre-compute per-node degree + color + parsed mtime.
  const graphData = useMemo(() => {
    if (!data) return null
    const degrees = new Map<string, number>()
    data.edges.forEach((e) => {
      degrees.set(e.source, (degrees.get(e.source) || 0) + 1)
      degrees.set(e.target, (degrees.get(e.target) || 0) + 1)
    })
    const maxDegree = Math.max(1, ...Array.from(degrees.values()))
    return {
      nodes: data.nodes.map((n) => ({
        ...n,
        mtime: n.created_at ? new Date(n.created_at).getTime() : 0,
        degree: degrees.get(n.id) || 0,
        color: getNodeColor(n.tags),
      })),
      links: data.edges,
      maxDegree,
      totalNodes: data.total_nodes,
      totalEdges: data.total_edges,
    }
  }, [data])

  // Time range (in epoch ms).
  const { timeMin, timeMax } = useMemo(() => {
    if (!graphData) return { timeMin: 0, timeMax: 0 }
    const times = graphData.nodes
      .map((n: any) => n.mtime)
      .filter((t: number) => t > 0)
    if (!times.length) return { timeMin: 0, timeMax: 0 }
    return { timeMin: Math.min(...times), timeMax: Math.max(...times) }
  }, [graphData])

  // Cutoff timestamp derived from the slider fraction.
  const cutoffTime = timeMin + (timeMax - timeMin) * timeFraction

  // 1-hop neighbor set of the selected node.
  const neighbors = useMemo(() => {
    const set = new Set<string>()
    if (!selectedNode || !graphData) return set
    set.add(selectedNode)
    graphData.links.forEach((l: any) => {
      const src = typeof l.source === "object" ? l.source.id : l.source
      const tgt = typeof l.target === "object" ? l.target.id : l.target
      if (src === selectedNode) set.add(tgt)
      if (tgt === selectedNode) set.add(src)
    })
    return set
  }, [selectedNode, graphData])

  // Legend entries actually present in the graph.
  const legendEntries = useMemo(() => {
    const present = new Set<string>()
    graphData?.nodes.forEach((n: any) => {
      const color = getNodeColor(n.tags)
      const tag = Object.keys(TAG_COLORS).find((k) => TAG_COLORS[k] === color)
      if (tag) present.add(tag)
    })
    return Object.keys(TAG_COLORS).filter((t) => present.has(t))
  }, [graphData])

  // Export the current canvas as a PNG snapshot.
  const handleExportPng = () => {
    const canvas = containerRef.current?.querySelector("canvas") as HTMLCanvasElement | null
    if (!canvas) return
    try {
      const url = canvas.toDataURL("image/png")
      const a = document.createElement("a")
      a.href = url
      a.download = `knowledge-graph-${new Date().toISOString().slice(0, 10)}.png`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
    } catch (e) {
      console.error("Failed to export PNG:", e)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex flex-col" style={{ backgroundColor: BG_COLOR }}>
      {/* Header */}
      <div
        className="border-b px-6 py-3 flex items-center justify-between shrink-0 backdrop-blur-md"
        style={{
          borderColor: "rgba(148,163,184,0.15)",
          backgroundColor: "rgba(10,14,39,0.85)",
        }}
      >
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="icon" onClick={onClose} className="shrink-0 text-slate-300 hover:bg-white/10" title="返回">
            <ArrowLeft className="w-5 h-5" />
          </Button>
          <GitFork className="w-5 h-5 text-blue-400 shrink-0" />
          <h2 className="text-lg font-semibold text-slate-100">知识图谱</h2>
          {graphData && (
            <span className="text-xs text-slate-400">
              {graphData.totalNodes} 节点 · {graphData.totalEdges} 关联
            </span>
          )}
          {selectedNode && (
            <span className="text-xs text-blue-300 bg-blue-500/20 px-2 py-0.5 rounded-full border border-blue-400/30">
              已选中：{selectedNode}
            </span>
          )}
        </div>
        <Button variant="ghost" size="icon" onClick={onClose} className="shrink-0 text-slate-300 hover:bg-white/10" title="关闭">
          <X className="w-5 h-5" />
        </Button>
      </div>
      <div className="flex items-center gap-2">
        <Button
          variant="ghost"
          size="sm"
          onClick={handleExportPng}
          className="text-slate-300 hover:bg-white/10 h-7 px-2.5 gap-1.5 text-xs shrink-0"
          title="导出 PNG"
        >
          <Download className="w-3.5 h-3.5" />
          导出 PNG
        </Button>
        <Button variant="ghost" size="icon" onClick={onClose} className="shrink-0 text-slate-300 hover:bg-white/10" title="关闭">
          <X className="w-5 h-5" />
        </Button>
      </div>

      {/* Time filter row */}
      {graphData && timeMin > 0 && timeMax > timeMin && (
        <div
          className="px-6 py-2 flex items-center gap-3 shrink-0 text-xs text-slate-400 border-b"
          style={{ borderColor: "rgba(148,163,184,0.1)" }}
        >
          <span className="font-mono">{formatDate(new Date(timeMin).toISOString())}</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={timeFraction}
            onChange={(e) => setTimeFraction(parseFloat(e.target.value))}
            className="flex-1 h-1.5 appearance-none rounded-full cursor-pointer"
            style={{
              background: `linear-gradient(to right, #60a5fa 0%, #60a5fa ${timeFraction * 100}%, rgba(148,163,184,0.25) ${timeFraction * 100}%, rgba(148,163,184,0.25) 100%)`,
            }}
          />
          <span className="font-mono">{formatDate(new Date(cutoffTime).toISOString())}</span>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setTimeFraction(1)}
            className="text-slate-300 hover:bg-white/10 h-6 px-2 text-xs"
          >
            重置
          </Button>
        </div>
      )}

      {/* Graph area */}
      <div className="flex-1 min-h-0 relative" ref={containerRef}>
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center text-slate-400">
            <Loader2 className="w-6 h-6 animate-spin" />
          </div>
        )}
        {error && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-sm text-red-400 gap-2">
            <p className="font-medium">加载图谱失败</p>
            <p className="text-xs text-slate-400">{error}</p>
          </div>
        )}
        {graphData && (
          <ForceGraph2D
            graphData={graphData}
            nodeLabel={(n: any) =>
              `${n.label}\n标签：${n.tags?.join("、") || "无"}\n${n.degree} 条关联\n创建：${formatDate(n.created_at)}`
            }
            nodeVal={(n: any) => 2 + (n.degree / graphData.maxDegree) * 6}
            nodeCanvasObject={(node: any, ctx: any, globalScale: number) => {
              const ratio = node.degree / graphData.maxDegree
              const outOfRange = node.mtime && cutoffTime > 0 && node.mtime > cutoffTime
              const isDimmed = !!selectedNode && !neighbors.has(node.id)

              // Effective alpha for time-filter / selection dimming.
              const alpha = outOfRange ? 0.06 : isDimmed ? 0.2 : 1

              // Radius scales with degree; hub nodes "radiate" more.
              const r = (3 + ratio * 7) / globalScale

              ctx.save()
              ctx.globalAlpha = alpha

              // Glow halo — blur grows with degree so hubs radiate energy.
              ctx.shadowColor = node.color
              ctx.shadowBlur = (8 + ratio * 24) / globalScale
              ctx.beginPath()
              ctx.arc(node.x, node.y, r, 0, 2 * Math.PI)
              ctx.fillStyle = node.color
              ctx.fill()
              ctx.shadowBlur = 0

              // Bright core dot (gives a "lit" center).
              ctx.beginPath()
              ctx.arc(node.x, node.y, r * 0.5, 0, 2 * Math.PI)
              ctx.fillStyle = "rgba(255, 255, 255, 0.9)"
              ctx.fill()

              ctx.restore()

              // Degree badge for hubs only.
              if (!outOfRange && !isDimmed && ratio > 0.3) {
                const fontSize = 9 / globalScale
                ctx.font = `${fontSize}px "JetBrains Mono", monospace`
                ctx.fillStyle = "rgba(255, 255, 255, 0.95)"
                ctx.textAlign = "center"
                ctx.textBaseline = "middle"
                ctx.fillText(String(node.degree), node.x, node.y - r - fontSize)
              }
            }}
            linkColor={(l: any) => {
              if (!selectedNode) return "rgba(148, 163, 184, 0.22)"
              const src = typeof l.source === "object" ? l.source.id : l.source
              const tgt = typeof l.target === "object" ? l.target.id : l.target
              return src === selectedNode || tgt === selectedNode
                ? "rgba(96, 165, 250, 0.55)"
                : "rgba(148, 163, 184, 0.05)"
            }}
            linkWidth={(l: any) => {
              if (!selectedNode) return 0.6
              const src = typeof l.source === "object" ? l.source.id : l.source
              const tgt = typeof l.target === "object" ? l.target.id : l.target
              return src === selectedNode || tgt === selectedNode ? 1.4 : 0.3
            }}
            onNodeClick={(n: any) => {
              setSelectedNode((prev) => (prev === n.id ? null : n.id))
            }}
            onBackgroundClick={() => setSelectedNode(null)}
            backgroundColor={BG_COLOR}
            cooldownTicks={100}
          />
        )}
      </div>

      {/* Legend footer */}
      <div
        className="px-6 py-2.5 shrink-0 flex items-center justify-between flex-wrap gap-2 border-t"
        style={{
          borderColor: "rgba(148,163,184,0.15)",
          backgroundColor: "rgba(10,14,39,0.6)",
        }}
      >
        <p className="text-xs text-slate-400">
          点击节点高亮关联；拖动时间滑块看笔记积累过程；点空白处取消。
        </p>
        <div className="flex items-center gap-3 flex-wrap">
          {legendEntries.map((tag) => (
            <span key={tag} className="inline-flex items-center gap-1.5 text-xs text-slate-300">
              <span
                className="h-2.5 w-2.5 rounded-full inline-block"
                style={{
                  backgroundColor: TAG_COLORS[tag],
                  boxShadow: `0 0 6px ${TAG_COLORS[tag]}80`,
                }}
              />
              {TAG_LABELS[tag] || tag}
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}