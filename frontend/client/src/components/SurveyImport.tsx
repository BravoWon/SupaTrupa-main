import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Upload, FileText, AlertCircle, CheckCircle } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

interface SurveyPoint {
  md: number;
  inc: number;
  azi: number;
  tvd: number;
  n_s: number;
  e_w: number;
  dls: number;
}

interface SurveyImportProps {
  onSurveyLoaded: (surveys: SurveyPoint[], type: 'plan' | 'actual' | 'offset') => void;
}

export function SurveyImport({ onSurveyLoaded }: SurveyImportProps) {
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const parseCSV = (text: string): SurveyPoint[] => {
    const lines = text.split('\n');
    const points: SurveyPoint[] = [];

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line || /^[a-zA-Z]/.test(line)) continue;

      const parts = line.split(/[, \t]+/).map(Number);
      if (parts.length >= 3 && !parts.some(isNaN)) {

        points.push({
          md: parts[0],
          inc: parts[1],
          azi: parts[2],
          tvd: parts[3] || 0,
          n_s: parts[4] || 0,
          e_w: parts[5] || 0,
          dls: parts[6] || 0
        });
      }
    }
    return points;
  };

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setError(null);
    setSuccess(null);

    const file = acceptedFiles[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      try {
        const text = reader.result as string;
        const points = parseCSV(text);

        if (points.length === 0) {
          setError("No valid survey data found. Check CSV format (MD, INC, AZI...).");
          return;
        }

        const type = file.name.toLowerCase().includes('plan') ? 'plan' : 'actual';

        onSurveyLoaded(points, type);
        setSuccess(`Successfully loaded ${points.length} survey points as ${type.toUpperCase()}.`);
      } catch (e) {
        setError("Failed to parse file. Ensure it is a valid CSV/Text file.");
      }
    };
    reader.readAsText(file);
  }, [onSurveyLoaded]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'text/plain': ['.txt', '.las', '.asc']
    },
    multiple: false
  });

  return (
    <Card className="border-border bg-card/30">
      <CardHeader className="pb-2 border-b border-border/50">
        <CardTitle className="text-sm font-mono flex items-center gap-2">
          <Upload className="w-4 h-4 text-chart-1" />
          SURVEY INGESTION
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-4">
        <div
          {...getRootProps()}
          className={`
            border-2 border-dashed rounded-md p-6 text-center cursor-pointer transition-colors
            ${isDragActive ? 'border-primary bg-primary/10' : 'border-muted-foreground/25 hover:border-primary/50'}
          `}
        >
          <input {...getInputProps()} />
          <FileText className="w-8 h-8 mx-auto mb-2 text-muted-foreground" />
          {isDragActive ? (
            <p className="text-sm text-primary font-medium">Drop the survey file here...</p>
          ) : (
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Drag & drop .CSV or .LAS files</p>
              <p className="text-xs text-muted-foreground/50">Supports MD, INC, AZI format</p>
            </div>
          )}
        </div>

        {error && (
          <Alert variant="destructive" className="mt-4 py-2">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {success && (
          <Alert className="mt-4 py-2 border-green-500/50 text-green-500">
            <CheckCircle className="h-4 w-4" />
            <AlertTitle>Success</AlertTitle>
            <AlertDescription>{success}</AlertDescription>
          </Alert>
        )}
      </CardContent>
    </Card>
  );
}
