"use client"

import { useMemo } from "react"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from "recharts"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { BarChart3 } from "lucide-react"

interface SpectralChartProps {
  data: number[][]
  headers: string[]
}

export function SpectralChart({ data, headers }: SpectralChartProps) {
  const chartData = useMemo(() => {
    if (!data || data.length === 0 || !headers || headers.length === 0) return []

    // Use the first row of data for visualization
    return headers.map((header, index) => ({
      wavelength: header,
      reflectance: data[0]?.[index] ?? 0,
    }))
  }, [data, headers])

  if (chartData.length === 0) {
    return (
      <Card className="border-border shadow-sm">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-foreground">
            <BarChart3 className="h-5 w-5 text-primary" />
            Spectral Visualization
          </CardTitle>
          <CardDescription>
            Wavelength vs. Reflectance plot of uploaded data
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center gap-3 py-12">
            <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-secondary">
              <BarChart3 className="h-7 w-7 text-muted-foreground" />
            </div>
            <div className="text-center">
              <p className="text-sm font-medium text-foreground">No spectral data</p>
              <p className="text-xs text-muted-foreground">
                Upload a CSV to view the spectral graph
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="border-border shadow-sm">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-foreground">
          <BarChart3 className="h-5 w-5 text-primary" />
          Spectral Visualization
        </CardTitle>
        <CardDescription>
          Wavelength vs. Reflectance ({data.length} sample{data.length > 1 ? "s" : ""} loaded)
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 8, right: 8, left: -10, bottom: 0 }}>
              <defs>
                <linearGradient id="spectralGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="oklch(0.55 0.2 255)" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="oklch(0.55 0.2 255)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.91 0.015 250)" />
              <XAxis
                dataKey="wavelength"
                tick={{ fontSize: 11, fill: "oklch(0.5 0.03 250)" }}
                axisLine={{ stroke: "oklch(0.91 0.015 250)" }}
                tickLine={{ stroke: "oklch(0.91 0.015 250)" }}
                label={{
                  value: "Wavelength",
                  position: "insideBottom",
                  offset: -2,
                  style: { fontSize: 12, fill: "oklch(0.5 0.03 250)" },
                }}
                interval={Math.max(0, Math.floor(chartData.length / 8))}
              />
              <YAxis
                tick={{ fontSize: 11, fill: "oklch(0.5 0.03 250)" }}
                axisLine={{ stroke: "oklch(0.91 0.015 250)" }}
                tickLine={{ stroke: "oklch(0.91 0.015 250)" }}
                label={{
                  value: "Reflectance",
                  angle: -90,
                  position: "insideLeft",
                  offset: 18,
                  style: { fontSize: 12, fill: "oklch(0.5 0.03 250)" },
                }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "oklch(1 0 0)",
                  border: "1px solid oklch(0.91 0.015 250)",
                  borderRadius: "0.75rem",
                  fontSize: "13px",
                  boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
                }}
                labelStyle={{ color: "oklch(0.18 0.02 250)", fontWeight: 600 }}
                itemStyle={{ color: "oklch(0.55 0.2 255)" }}
              />
              <Area
                type="monotone"
                dataKey="reflectance"
                stroke="oklch(0.55 0.2 255)"
                strokeWidth={2.5}
                fill="url(#spectralGradient)"
                dot={false}
                activeDot={{ r: 5, fill: "oklch(0.55 0.2 255)", stroke: "#fff", strokeWidth: 2 }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  )
}
