import { Salad, Sparkles, ChevronLeft } from "lucide-react"
import Link from "next/link"

function PageShell({
  title,
  subtitle,
  accent = "teal",
  children,
}: {
  title: string
  subtitle: string
  accent?: "emerald" | "amber" | "teal"
  children: React.ReactNode
}) {
  const accentClass =
    accent === "amber"
      ? "from-amber-500 via-orange-500 to-yellow-500"
      : accent === "teal"
        ? "from-teal-500 via-cyan-500 to-emerald-500"
        : "from-emerald-500 via-lime-500 to-teal-500"

  const softClass =
    accent === "amber"
      ? "bg-amber-50 text-amber-700 border-amber-200/80"
      : accent === "teal"
        ? "bg-teal-50 text-teal-700 border-teal-200/80"
        : "bg-emerald-50 text-emerald-700 border-emerald-200/80"

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(240,253,250,0.96),_transparent_32%),linear-gradient(180deg,#fbfefc_0%,#eef7f1_100%)] text-foreground">
      <div className="mx-auto max-w-6xl px-4 py-6 sm:px-6 lg:px-8">
        <div className="mb-6 flex items-center justify-between rounded-3xl border border-white/80 bg-white/70 px-4 py-3 shadow-sm backdrop-blur-md">
          <div className="flex items-center gap-3">
            <div className={`flex h-10 w-10 items-center justify-center rounded-2xl bg-gradient-to-br shadow-[0_12px_30px_rgba(20,184,166,0.16)] ring-1 ring-white/70 ${accentClass}`}>
              <Salad className="h-5 w-5 text-white" />
            </div>
            <div>
              <div className="text-sm font-semibold text-slate-800">个人知识管理助手</div>
              <div className="text-xs text-slate-500">让饮食建议更清楚、更可执行</div>
            </div>
          </div>
          <Link href="/" className="inline-flex items-center gap-2 rounded-full border border-border/60 bg-white px-3 py-2 text-sm text-slate-600 shadow-sm hover:bg-slate-50">
            <ChevronLeft className="h-4 w-4" />
            返回首页
          </Link>
        </div>

        <div className="rounded-[28px] border border-border/60 bg-white/85 p-6 shadow-[0_20px_60px_rgba(15,23,42,0.06)] backdrop-blur-sm sm:p-8">
          <div className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-medium ${softClass}`}>
            <Sparkles className="h-3.5 w-3.5" />
            FAQ 页面
          </div>
          <h1 className="mt-4 text-2xl font-semibold tracking-tight sm:text-3xl">{title}</h1>
          <p className="mt-3 max-w-3xl text-sm leading-7 text-muted-foreground">{subtitle}</p>
          <div className="mt-8">{children}</div>
        </div>
      </div>
    </div>
  )
}

export default function FAQPage() {
  return (
    <PageShell
      title="常见问题"
      subtitle="这里整理常见提问方式，方便你快速开始，不用每次从零描述。"
      accent="teal"
    >
      <div className="grid gap-4 md:grid-cols-2">
        <section className="rounded-2xl border border-teal-100 bg-teal-50/70 p-5">
          <h2 className="text-base font-semibold">你可以直接这样问</h2>
          <ul className="mt-3 space-y-2 text-sm leading-6 text-muted-foreground">
            <li>• “我想减脂，但晚上容易饿，怎么安排？”</li>
            <li>• “食堂午餐怎么搭配更均衡？”</li>
            <li>• “帮我列 3 个低成本高蛋白加餐。”</li>
          </ul>
        </section>

        <section className="rounded-2xl border border-border/60 bg-slate-50/80 p-5">
          <h2 className="text-base font-semibold">提问小技巧</h2>
          <ul className="mt-3 space-y-2 text-sm leading-6 text-muted-foreground">
            <li>• 先说目标，再说限制</li>
            <li>• 如果是给自己用，补充年龄、身高、体重、活动量</li>
            <li>• 如果是具体场景，说明时间、地点和预算</li>
          </ul>
        </section>
      </div>
    </PageShell>
  )
}
