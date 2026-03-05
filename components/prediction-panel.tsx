"use client"

import { CheckCircle2, AlertCircle, Beaker, TrendingUp } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"

export interface PredictionResult {
  fabricType: string
  confidence: number
  status: "success" | "error"
  message?: string
}

interface PredictionPanelProps {
  result: PredictionResult | null
  isLoading: boolean
}

const fabricDetails: Record<string, { description: string; color: string }> = {
  Cotton: {
    description: "Natural plant fiber with excellent breathability and moisture absorption.",
    color: "bg-chart-3 text-accent-foreground",
  },
  "Cotton-Poly Blend": {
    description: "Hybrid textile combining natural comfort with synthetic durability.",
    color: "bg-chart-2 text-accent-foreground",
  },
  "Poly-Spandex": {
    description: "Synthetic stretch fabric with high elasticity and shape retention.",
    color: "bg-chart-1 text-primary-foreground",
  },
}

export function PredictionPanel({ result, isLoading }: PredictionPanelProps) {
  return (
    <Card className="border-border shadow-sm">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-foreground">
          <Beaker className="h-5 w-5 text-primary" />
          Prediction Results
        </CardTitle>
        <CardDescription>
          AI classification output and confidence metrics
        </CardDescription>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="flex flex-col items-center gap-4 py-8">
            <div className="relative flex h-16 w-16 items-center justify-center">
              <div className="absolute h-16 w-16 animate-spin rounded-full border-4 border-secondary border-t-primary" />
              <Beaker className="h-6 w-6 text-primary" />
            </div>
            <div className="text-center">
              <p className="text-sm font-medium text-foreground">Analyzing spectral data...</p>
              <p className="text-xs text-muted-foreground">Running classification model</p>
            </div>
          </div>
        ) : result ? (
          <div className="flex flex-col gap-5">
            <div className="flex items-start gap-3">
              {result.status === "success" ? (
                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-chart-3/20">
                  <CheckCircle2 className="h-5 w-5 text-chart-3" />
                </div>
              ) : (
                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-destructive/10">
                  <AlertCircle className="h-5 w-5 text-destructive" />
                </div>
              )}
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <Badge
                    variant={result.status === "success" ? "default" : "destructive"}
                    className={result.status === "success" ? "bg-chart-3/20 text-chart-3 border border-chart-3/30" : ""}
                  >
                    {result.status === "success" ? "Classification Complete" : "Error"}
                  </Badge>
                </div>
                {result.status === "error" && result.message && (
                  <p className="mt-1 text-sm text-muted-foreground">{result.message}</p>
                )}
              </div>
            </div>

            {result.status === "success" && (
              <>
                <div className="rounded-xl bg-secondary/70 p-5">
                  <p className="mb-1 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                    Identified Fabric
                  </p>
                  <p className="text-2xl font-bold text-foreground">{result.fabricType}</p>
                  {fabricDetails[result.fabricType] && (
                    <p className="mt-1.5 text-sm text-muted-foreground leading-relaxed">
                      {fabricDetails[result.fabricType].description}
                    </p>
                  )}
                </div>

                <div className="flex flex-col gap-2.5">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1.5">
                      <TrendingUp className="h-4 w-4 text-primary" />
                      <span className="text-sm font-medium text-foreground">Confidence</span>
                    </div>
                    <span className="text-lg font-bold font-mono text-primary">
                      {(result.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <Progress value={result.confidence * 100} className="h-2.5" />
                  <p className="text-xs text-muted-foreground">
                    {result.confidence > 0.9
                      ? "Very high confidence classification"
                      : result.confidence > 0.7
                        ? "Good confidence - results are reliable"
                        : "Low confidence - consider re-running with cleaner data"}
                  </p>
                </div>
              </>
            )}
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3 py-8">
            <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-secondary">
              <Beaker className="h-7 w-7 text-muted-foreground" />
            </div>
            <div className="text-center">
              <p className="text-sm font-medium text-foreground">No results yet</p>
              <p className="text-xs text-muted-foreground">
                Upload a CSV file and run prediction
              </p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
