import { Salad, Sparkles, ChevronLeft } from "lucide-react"
import Link from "next/link"

function PageShell({
  title,
  subtitle,
  accent = "emerald",
  children,
}: {
  title: string
  subtitle: string
  accent?: "emerald" | "amber" | "teal"
  children: React.ReactNode
}) {
  const accentStyles =
    accent === "amber"
      ? "from-amber-500 via-orange-500 to-yellow-500 text-amber-700 bg-amber-50 border-amber-200/80"
      : accent === "teal"
        ? "from-teal-500 via-cyan-500 to-emerald-500 text-teal-700 bg-teal-50 border-teal-200/80"
        : "from-emerald-500 via-lime-500 to-teal-500 text-emerald-700 bg-emerald-50 border-emerald-200/80"

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(236,253,245,0.95),_transparent_32%),linear-gradient(180deg,#fbfefc_0%,#eef7f1_100%)] text-foreground">
      <div className="mx-auto max-w-6xl px-4 py-6 sm:px-6 lg:px-8">
        <div className="mb-6 flex items-center justify-between rounded-3xl border border-white/80 bg-white/70 px-4 py-3 shadow-sm backdrop-blur-md">
          <div className="flex items-center gap-3">
            <div className={`flex h-10 w-10 items-center justify-center rounded-2xl bg-gradient-to-br shadow-[0_12px_30px_rgba(16,185,129,0.16)] ring-1 ring-white/70 ${accentStyles.split(" ").slice(0, 3).join(" ")}`}>
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
          <div className="inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-medium ${accentStyles}">
            <Sparkles className="h-3.5 w-3.5" />
            指南页面
          </div>
          <h1 className="mt-4 text-2xl font-semibold tracking-tight sm:text-3xl">{title}</h1>
          <p className="mt-3 max-w-3xl text-sm leading-7 text-muted-foreground">{subtitle}</p>
          <div className="mt-8">{children}</div>
        </div>
      </div>
    </div>
  )
}

export default function GuidePage() {
  return (
    <PageShell
      title="饮食指南"
      subtitle="提问时尽量把目标、限制和场景说清楚，这样更容易得到可执行的建议。"
    >
      <div className="grid gap-4 md:grid-cols-2">
        <section className="rounded-2xl border border-emerald-100 bg-emerald-50/70 p-5">
          <h2 className="text-base font-semibold">建议包含的信息</h2>
          <ul className="mt-3 space-y-2 text-sm leading-6 text-muted-foreground">
            <li>• 目标：减脂、增肌、控糖、改善胃口、提高饱腹感</li>
            <li>• 条件：身高体重、作息、预算、忌口、过敏史</li>
            <li>• 场景：早餐、午餐、晚餐、外卖、食堂、夜宵</li>
          </ul>
        </section>

        <section className="rounded-2xl border border-border/60 bg-slate-50/80 p-5">
          <h2 className="text-base font-semibold">可直接这样问</h2>
          <ul className="mt-3 space-y-2 text-sm leading-6 text-muted-foreground">
            <li>• 我想减脂，但晚上容易饿，怎么安排？</li>
            <li>• 帮我做一份 7 天早餐搭配，要求高蛋白低负担。</li>
            <li>• 学生党食堂午餐怎么选更均衡？</li>
          </ul>
        </section>
      </div>
    </PageShell>
  )
}
