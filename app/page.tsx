"use client"

import { useCallback, useState } from "react"
import { Header } from "@/components/header"
import { FileUpload } from "@/components/file-upload"
import { PredictionPanel, type PredictionResult } from "@/components/prediction-panel"
import { SpectralChart } from "@/components/spectral-chart"
import { Instructions } from "@/components/instructions"

export default function Home() {
  const [csvData, setCsvData] = useState<number[][]>([])
  const [csvHeaders, setCsvHeaders] = useState<string[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<PredictionResult | null>(null)

  const handleFileUpload = useCallback((data: number[][], headers: string[]) => {
    setCsvData(data)
    setCsvHeaders(headers)
    setResult(null)
  }, [])

  const handlePredict = useCallback(() => {
    if (csvData.length === 0) return

    setIsLoading(true)
    setResult(null)

    // Simulate ML model inference with realistic delay
    setTimeout(() => {
      try {
        // Simple mock classification based on spectral data characteristics
        const row = csvData[0]
        const mean = row.reduce((a, b) => a + b, 0) / row.length
        const max = Math.max(...row)

        const fabricTypes = ["Cotton", "Cotton-Poly Blend", "Poly-Spandex"]
        let typeIndex: number
        let confidence: number

        if (mean < 0.3) {
          typeIndex = 0
          confidence = 0.87 + Math.random() * 0.1
        } else if (mean < 0.6) {
          typeIndex = 1
          confidence = 0.82 + Math.random() * 0.12
        } else {
          typeIndex = 2
          confidence = 0.9 + Math.random() * 0.08
        }

        setResult({
          fabricType: fabricTypes[typeIndex],
          confidence,
          status: "success",
        })
      } catch {
        setResult({
          fabricType: "",
          confidence: 0,
          status: "error",
          message: "An error occurred during classification. Please check your data format.",
        })
      }
      setIsLoading(false)
    }, 2200)
  }, [csvData])

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
        <div className="grid gap-6 lg:grid-cols-12">
          {/* Left column - Upload & Instructions */}
          <div className="flex flex-col gap-6 lg:col-span-4">
            <FileUpload
              onFileUpload={handleFileUpload}
              onPredict={handlePredict}
              isLoading={isLoading}
              hasFile={csvData.length > 0}
            />
            <Instructions />
          </div>

          {/* Right column - Results & Chart */}
          <div className="flex flex-col gap-6 lg:col-span-8">
            <PredictionPanel result={result} isLoading={isLoading} />
            <SpectralChart data={csvData} headers={csvHeaders} />
          </div>
        </div>
      </main>
    </div>
  )
}
