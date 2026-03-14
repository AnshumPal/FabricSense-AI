"use client"

import { useCallback, useState } from "react"
import { Header } from "@/components/header"
import { FileUpload } from "@/components/file-upload"
import { PredictionPanel, type PredictionResult } from "@/components/prediction-panel"
import { SpectralChart } from "@/components/spectral-chart"
import { Instructions } from "@/components/instructions"

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

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

  const handlePredict = useCallback(async () => {
    if (csvData.length === 0) return

    setIsLoading(true)
    setResult(null)

    try {
      // Reconstruct CSV from parsed data and send to backend
      const csvContent = `${csvHeaders.join(",")}\n${csvData.map((row) => row.join(",")).join("\n")}`
      const csvBlob = new Blob([csvContent], { type: "text/csv" })
      const formData = new FormData()
      formData.append("file", csvBlob, "spectral_data.csv")

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      })

      const responseText = await response.text()

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${responseText}`)
      }

      const apiResult = JSON.parse(responseText)

      // Map backend response to frontend fabric type labels
      const fabricTypeMap: Record<string, string> = {
        Cotton: "Cotton",
        "Cotton/Poly blend": "Cotton-Poly Blend",
        "Poly/Spandex": "Poly-Spandex",
      }

      setResult({
        fabricType: fabricTypeMap[apiResult.predicted_fabric] ?? apiResult.predicted_fabric ?? "Unknown",
        confidence: parseFloat(apiResult.confidence) || 0,
        status: "success",
      })
    } catch (error) {
      console.error("Prediction error:", error)
      setResult({
        fabricType: "",
        confidence: 0,
        status: "error",
        message: error instanceof Error ? `Prediction failed: ${error.message}` : "An unexpected error occurred.",
      })
    } finally {
      setIsLoading(false)
    }
  }, [csvData, csvHeaders])


  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
        <div className="grid gap-6 lg:grid-cols-12">
          {/* Left column - Upload & Instructions */}
          <div className="flex flex-col gap-6 lg:col-span-4">
            <FileUpload
              onFileUpload={handleFileUpload}
              onPredict={handlePredict}  // ✅ Matches updated FileUpload
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
