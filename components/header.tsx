import { Activity } from "lucide-react"

export function Header() {
  return (
    <header className="border-b border-border bg-card">
      <div className="mx-auto flex max-w-7xl items-center gap-3 px-6 py-5">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary">
            <Activity className="h-5 w-5 text-primary-foreground" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-foreground">
              FabricSense AI
            </h1>
            <p className="text-sm text-muted-foreground">
              Hyperspectral Textile Classification System
            </p>
          </div>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <span className="inline-flex items-center gap-1.5 rounded-full bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
            <span className="h-1.5 w-1.5 rounded-full bg-primary animate-pulse" />
            Model Ready
          </span>
        </div>
      </div>
    </header>
  )
}
