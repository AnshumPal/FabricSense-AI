"use client"

import { useCallback, useState } from "react"
import { Upload, FileText, X } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"

interface FileUploadProps {
  onFileUpload: (data: number[][], headers: string[]) => void
  onPredict: () => void
  isLoading: boolean
  hasFile: boolean
}

export function FileUpload({ onFileUpload, onPredict, isLoading, hasFile }: FileUploadProps) {
  const [fileName, setFileName] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isDragOver, setIsDragOver] = useState(false)

  const parseCSV = useCallback(
    (file: File) => {
      setError(null)
      const reader = new FileReader()
      reader.onload = (e) => {
        try {
          const text = e.target?.result as string
          const lines = text.trim().split("\n")
          if (lines.length < 2) {
            setError("CSV must have a header row and at least one data row.")
            return
          }
          const headers = lines[0].split(",").map((h) => h.trim())
          const data: number[][] = []
          for (let i = 1; i < lines.length; i++) {
            const values = lines[i].split(",").map((v) => {
              const n = parseFloat(v.trim())
              if (isNaN(n)) throw new Error(`Non-numeric value at row ${i + 1}`)
              return n
            })
            if (values.length !== headers.length) {
              throw new Error(`Row ${i + 1} has ${values.length} columns but header has ${headers.length}`)
            }
            data.push(values)
          }
          setFileName(file.name)
          onFileUpload(data, headers)
        } catch (err) {
          setError(err instanceof Error ? err.message : "Failed to parse CSV file.")
          setFileName(null)
        }
      }
      reader.readAsText(file)
    },
    [onFileUpload]
  )

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragOver(false)
      const file = e.dataTransfer.files[0]
      if (file && file.name.endsWith(".csv")) {
        parseCSV(file)
      } else {
        setError("Please upload a .csv file only.")
      }
    },
    [parseCSV]
  )

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file && file.name.endsWith(".csv")) {
        parseCSV(file)
      } else if (file) {
        setError("Please upload a .csv file only.")
      }
    },
    [parseCSV]
  )

  const handleRemoveFile = () => {
    setFileName(null)
    setError(null)
  }

  return (
    <Card className="border-border shadow-sm">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-foreground">
          <Upload className="h-5 w-5 text-primary" />
          Upload Spectral Data
        </CardTitle>
        <CardDescription>
          Drag and drop a CSV file or click to browse
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <div
          onDrop={handleDrop}
          onDragOver={(e) => {
            e.preventDefault()
            setIsDragOver(true)
          }}
          onDragLeave={() => setIsDragOver(false)}
          className={`relative flex min-h-40 cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed transition-all ${
            isDragOver
              ? "border-primary bg-primary/5"
              : fileName
                ? "border-primary/40 bg-primary/5"
                : "border-border bg-secondary/50 hover:border-primary/40 hover:bg-primary/5"
          }`}
        >
          <input
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="absolute inset-0 cursor-pointer opacity-0"
            aria-label="Upload CSV file"
          />
          {fileName ? (
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                <FileText className="h-5 w-5 text-primary" />
              </div>
              <div>
                <p className="text-sm font-medium text-foreground">{fileName}</p>
                <p className="text-xs text-muted-foreground">Ready for prediction</p>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  handleRemoveFile()
                }}
                className="ml-2 rounded-full p-1 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
                aria-label="Remove file"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-2 px-4 py-2">
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10">
                <Upload className="h-6 w-6 text-primary" />
              </div>
              <p className="text-sm font-medium text-foreground">
                Drop your CSV file here
              </p>
              <p className="text-xs text-muted-foreground">or click to browse</p>
            </div>
          )}
        </div>

        {error && (
          <div className="rounded-lg bg-destructive/10 px-4 py-3 text-sm text-destructive">
            {error}
          </div>
        )}

        <Button
          onClick={onPredict}
          disabled={!hasFile || isLoading}
          className="w-full"
          size="lg"
        >
          {isLoading ? (
            <span className="flex items-center gap-2">
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-primary-foreground border-t-transparent" />
              Analyzing...
            </span>
          ) : (
            "Predict Fabric Type"
          )}
        </Button>
      </CardContent>
    </Card>
  )
}
