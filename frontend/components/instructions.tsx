import { Info } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export function Instructions() {
  return (
    <Card className="border-border shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-sm text-foreground">
          <Info className="h-4 w-4 text-primary" />
          CSV Format Guide
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col gap-3 text-sm text-muted-foreground leading-relaxed">
          <p>
            Upload a CSV file containing hyperspectral reflectance data. The file should be structured as:
          </p>
          <div className="overflow-hidden rounded-lg border border-border">
            <div className="bg-secondary/70 px-3 py-1.5 text-xs font-medium text-muted-foreground">
              example.csv
            </div>
            <pre className="overflow-x-auto bg-secondary/30 px-3 py-2.5 font-mono text-xs text-foreground">
              <code>{"400,410,420,...,2500\n0.12,0.15,0.18,...,0.34"}</code>
            </pre>
          </div>
          <ul className="flex flex-col gap-1.5 pl-4">
            <li className="list-disc">
              <span className="font-medium text-foreground">Header row:</span> Wavelength values (in nm)
            </li>
            <li className="list-disc">
              <span className="font-medium text-foreground">Data rows:</span> Corresponding reflectance values (0-1)
            </li>
            <li className="list-disc">
              Supported fabric types: Cotton, Cotton-Poly Blend, Poly-Spandex
            </li>
          </ul>
        </div>
      </CardContent>
    </Card>
  )
}
