"use client"

import { useMemo, useState } from "react"
import { FileText, Tag, Loader2, ExternalLink, GitFork, X, ArrowLeft } from "lucide-react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { useNotes, type NoteDetail, type NoteGraph } from "@/lib/hooks/notes/use-notes"
import { GraphView } from "./graph-view"

interface NotesPanelProps {
  onSelectNote?: (noteName: string) => void
}

// Convert [[target]] and [[target|alias]] into internal note:// links so
// ReactMarkdown renders them as clickable jumps.
function preprocessWikilinks(content: string): string {
  return content.replace(/\[\[([^\]|]+)(?:\|([^\]]+))?\]\]/g, (_m, target, alias) => {
    const label = alias || target
    return `[${label}](note://${target})`
  })
}

export function NotesPanel({ onSelectNote }: NotesPanelProps) {
  const { notes, isLoading, error, getNote, getNoteGraph } = useNotes()
  const [activeTag, setActiveTag] = useState<string | null>(null)
  const [selectedNote, setSelectedNote] = useState<NoteDetail | null>(null)
  const [noteGraph, setNoteGraph] = useState<NoteGraph | null>(null)
  const [loadingDetail, setLoadingDetail] = useState(false)
  const [showGraph, setShowGraph] = useState(false)

  const allTags = useMemo(() => {
    const set = new Set<string>()
    notes.forEach((n) => n.tags.forEach((t) => set.add(t)))
    return Array.from(set).sort()
  }, [notes])

  const filteredNotes = useMemo(() => {
    if (!activeTag) return notes
    return notes.filter((n) => n.tags.includes(activeTag))
  }, [notes, activeTag])

  const handleOpenNote = async (name: string) => {
    setLoadingDetail(true)
    setSelectedNote(null)
    setNoteGraph(null)
    const [detail, graph] = await Promise.all([getNote(name), getNoteGraph(name)])
    setSelectedNote(detail)
    setNoteGraph(graph)
    setLoadingDetail(false)
    onSelectNote?.(name)
  }

  const handleCloseNote = () => {
    setSelectedNote(null)
    setNoteGraph(null)
  }

  // Markdown renderers for note content
  const markdownComponents = useMemo(() => ({
    a: ({ href, children, ...props }: any) => {
      if (href?.startsWith("note://")) {
        const target = decodeURIComponent(href.replace("note://", ""))
        return (
          <button
            type="button"
            onClick={() => handleOpenNote(target)}
            className="inline-flex items-center gap-0.5 px-1 mx-0.5 rounded bg-blue-50 text-blue-700 font-medium cursor-pointer hover:bg-blue-100 hover:underline transition-colors"
          >
            <FileText className="w-3 h-3" />
            {children}
          </button>
        )
      }
      return (
        <a {...props} href={href} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
          {children}
        </a>
      )
    },
    h1: ({ children }: any) => (
      <h1 className="text-2xl font-semibold mt-8 mb-4 text-foreground border-b border-border/60 pb-2">{children}</h1>
    ),
    h2: ({ children }: any) => (
      <h2 className="text-xl font-semibold mt-7 mb-3 text-foreground">{children}</h2>
    ),
    h3: ({ children }: any) => (
      <h3 className="text-lg font-medium mt-6 mb-2 text-foreground">{children}</h3>
    ),
    p: ({ children }: any) => (
      <p className="my-3 leading-7 text-[15px]">{children}</p>
    ),
    ul: ({ children }: any) => (
      <ul className="my-3 pl-6 space-y-1.5 list-disc text-[15px]">{children}</ul>
    ),
    ol: ({ children }: any) => (
      <ol className="my-3 pl-6 space-y-1.5 list-decimal text-[15px]">{children}</ol>
    ),
    li: ({ children }: any) => (
      <li className="leading-7">{children}</li>
    ),
    blockquote: ({ children }: any) => (
      <blockquote className="my-3 pl-4 border-l-4 border-blue-300 text-muted-foreground italic">{children}</blockquote>
    ),
    code: ({ className, children, ...props }: any) => {
      const isInline = !className && !String(children).includes("\n")
      if (isInline) {
        return (
          <code className="px-1.5 py-0.5 rounded bg-muted text-[13px] font-mono text-foreground" {...props}>
            {children}
          </code>
        )
      }
      return (
        <pre className="my-4 p-4 rounded-lg bg-slate-900 text-slate-100 overflow-x-auto text-[13px] leading-6 font-mono">
          <code className={className} {...props}>{children}</code>
        </pre>
      )
    },
    table: ({ children }: any) => (
      <div className="my-4 overflow-x-auto rounded-lg border border-border/60">
        <table className="w-full text-sm border-collapse">{children}</table>
      </div>
    ),
    th: ({ children }: any) => (
      <th className="px-3 py-2 text-left font-medium bg-muted border-b border-border/60">{children}</th>
    ),
    td: ({ children }: any) => (
      <td className="px-3 py-2 border-b border-border/40">{children}</td>
    ),
    hr: () => <hr className="my-6 border-border/60" />,
  }), [])

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center py-10 gap-2 text-muted-foreground">
        <Loader2 className="w-5 h-5 animate-spin" />
        <p className="text-xs">正在加载笔记...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="px-4 py-8 text-center text-sm text-destructive">
        <p className="font-medium mb-1">加载笔记失败</p>
        <p className="text-xs text-muted-foreground">{error}</p>
      </div>
    )
  }

  return (
    <>
      {/* Default view: tag filter + note list */}
      <div className="flex flex-col h-full min-h-0">
        {/* Knowledge graph entry */}
        <div className="px-3 pt-2 shrink-0">
          <button
            type="button"
            onClick={() => setShowGraph(true)}
            className="w-full flex items-center gap-2.5 px-3 py-2.5 rounded-lg border border-blue-100 bg-gradient-to-r from-blue-50 to-indigo-50 hover:from-blue-100 hover:to-indigo-100 transition-all group"
          >
            <GitFork className="w-4 h-4 text-blue-600 shrink-0" />
            <div className="flex-1 text-left">
              <div className="text-xs font-medium text-blue-700">知识图谱</div>
              <div className="text-[10px] text-blue-500/70">查看笔记间的 [[wikilink]] 关联</div>
            </div>
          </button>
        </div>

        <div className="px-3 py-2 border-b border-border/60 shrink-0">
          <div className="flex flex-wrap gap-1.5">
            <button
              type="button"
              onClick={() => setActiveTag(null)}
              className={`px-2 py-1 rounded-full text-[11px] font-medium transition-colors ${
                activeTag === null
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted text-muted-foreground hover:bg-muted/80"
              }`}
            >
              全部 ({notes.length})
            </button>
            {allTags.map((tag) => (
              <button
                key={tag}
                type="button"
                onClick={() => setActiveTag(activeTag === tag ? null : tag)}
                className={`px-2 py-1 rounded-full text-[11px] font-medium transition-colors ${
                  activeTag === tag
                    ? "bg-primary text-primary-foreground"
                    : "bg-muted text-muted-foreground hover:bg-muted/80"
                }`}
              >
                #{tag}
              </button>
            ))}
          </div>
        </div>

        <ScrollArea type="always" className="flex-1 min-h-0">
          <div className="p-2 space-y-1">
            {filteredNotes.length === 0 ? (
              <div className="px-4 py-8 text-center text-xs text-muted-foreground">
                没有匹配的笔记
              </div>
            ) : (
              filteredNotes.map((note) => (
                <button
                  key={note.name}
                  type="button"
                  onClick={() => handleOpenNote(note.name)}
                  className="w-full flex items-start gap-2 px-3 py-2.5 rounded-lg text-left transition-colors hover:bg-muted/60 group"
                >
                  <FileText className="w-4 h-4 mt-0.5 shrink-0 text-primary/70" />
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium text-foreground truncate group-hover:text-primary">
                      {note.title}
                    </div>
                    <div className="flex items-center gap-2 mt-0.5 text-[10px] text-muted-foreground">
                      {note.tags.length > 0 && (
                        <span className="inline-flex items-center gap-0.5">
                          <Tag className="w-3 h-3" />
                          {note.tags.slice(0, 3).join(" ")}
                        </span>
                      )}
                      {note.wikilink_count > 0 && (
                        <span className="inline-flex items-center gap-0.5">
                          <GitFork className="w-3 h-3" />
                          {note.wikilink_count}
                        </span>
                      )}
                    </div>
                  </div>
                </button>
              ))
            )}
          </div>
        </ScrollArea>
      </div>

      {/* Fullscreen note detail */}
      {selectedNote && (
        <div className="fixed inset-0 z-50 bg-background flex flex-col">
          {/* Header */}
          <div className="border-b border-border/60 px-6 py-3 flex items-center justify-between bg-background/95 backdrop-blur-sm shrink-0">
            <div className="flex items-center gap-3 min-w-0">
              <Button
                variant="ghost"
                size="icon"
                onClick={handleCloseNote}
                className="shrink-0"
                title="返回笔记列表"
              >
                <ArrowLeft className="w-5 h-5" />
              </Button>
              <FileText className="w-5 h-5 text-primary shrink-0" />
              <h2 className="text-lg font-semibold truncate">
                {selectedNote.title}
              </h2>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={handleCloseNote}
              className="shrink-0"
              title="关闭"
            >
              <X className="w-5 h-5" />
            </Button>
          </div>

          {/* Tags */}
          {selectedNote.tags.length > 0 && (
            <div className="px-6 py-2 border-b border-border/60 flex flex-wrap gap-1.5 shrink-0">
              {selectedNote.tags.map((tag) => (
                <span
                  key={tag}
                  className="px-2 py-0.5 rounded-full text-[11px] bg-primary/10 text-primary"
                >
                  #{tag}
                </span>
              ))}
            </div>
          )}

          {/* Content - scrollable */}
          <ScrollArea type="always" className="flex-1 min-h-0">
            <div className="max-w-4xl mx-auto px-8 py-8">
              {loadingDetail ? (
                <div className="flex items-center justify-center py-20 text-muted-foreground">
                  <Loader2 className="w-5 h-5 animate-spin" />
                </div>
              ) : (
                <article className="text-foreground">
                  <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                    {preprocessWikilinks(selectedNote.content)}
                  </ReactMarkdown>
                </article>
              )}
            </div>
          </ScrollArea>

          {/* Wikilinks footer */}
          {noteGraph && (noteGraph.out_links.length > 0 || noteGraph.in_links.length > 0) && (
            <div className="border-t border-border/60 px-6 py-3 bg-muted/30 shrink-0">
              <div className="max-w-4xl mx-auto">
                <div className="font-medium text-foreground text-xs mb-2 flex items-center gap-1.5">
                  <GitFork className="w-3.5 h-3.5 text-primary" />
                  关联笔记
                </div>
                {noteGraph.out_links.length > 0 && (
                  <div className="mb-1.5 flex items-center flex-wrap gap-1.5">
                    <span className="text-xs font-medium text-foreground/70">链出：</span>
                    {noteGraph.out_links.map((link) => (
                      <button
                        key={link}
                        type="button"
                        onClick={() => handleOpenNote(link)}
                        className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-xs bg-blue-50 text-blue-700 cursor-pointer hover:bg-blue-100 transition-colors"
                      >
                        <ExternalLink className="w-3 h-3" />
                        {link}
                      </button>
                    ))}
                  </div>
                )}
                {noteGraph.in_links.length > 0 && (
                  <div className="flex items-center flex-wrap gap-1.5">
                    <span className="text-xs font-medium text-foreground/70">链入：</span>
                    {noteGraph.in_links.map((link) => (
                      <button
                        key={link}
                        type="button"
                        onClick={() => handleOpenNote(link)}
                        className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-xs bg-teal-50 text-teal-700 cursor-pointer hover:bg-teal-100 transition-colors"
                      >
                        <ExternalLink className="w-3 h-3" />
                        {link}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Fullscreen knowledge graph */}
      {showGraph && (
        <GraphView
          onClose={() => setShowGraph(false)}
          onOpenNote={(name) => {
            setShowGraph(false)
            handleOpenNote(name)
          }}
        />
      )}
    </>
  )
}
