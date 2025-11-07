import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { AlertCircle, CheckCircle2, AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";

interface ResultsCardProps {
  probability: number;
  riskLevel: "low" | "moderate" | "high";
}

export const ResultsCard = ({ probability, riskLevel }: ResultsCardProps) => {
  const getRiskColor = () => {
    switch (riskLevel) {
      case "low":
        return "text-success";
      case "moderate":
        return "text-warning";
      case "high":
        return "text-destructive";
    }
  };

  const getRiskIcon = () => {
    switch (riskLevel) {
      case "low":
        return <CheckCircle2 className="h-8 w-8 text-success" />;
      case "moderate":
        return <AlertTriangle className="h-8 w-8 text-warning" />;
      case "high":
        return <AlertCircle className="h-8 w-8 text-destructive" />;
    }
  };

  const getRiskMessage = () => {
    switch (riskLevel) {
      case "low":
        return "Your diabetes risk appears to be low based on the provided information.";
      case "moderate":
        return "You show moderate risk factors for diabetes. Consider consulting a healthcare professional.";
      case "high":
        return "You have significant risk factors for diabetes. We strongly recommend consulting a healthcare professional.";
    }
  };

  return (
    <Card className="border-2 animate-slide-up" style={{ boxShadow: "var(--shadow-lg)" }}>
      <CardHeader>
        <div className="flex items-center gap-3">
          {getRiskIcon()}
          <div>
            <CardTitle className="text-2xl">Assessment Results</CardTitle>
            <CardDescription>Based on your responses</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <span className="text-sm font-medium">Diabetes Risk Probability</span>
            <span className={cn("text-2xl font-bold", getRiskColor())}>
              {(probability * 100).toFixed(1)}%
            </span>
          </div>
          <Progress value={probability * 100} className="h-3" />
        </div>

        <div className="p-4 rounded-lg bg-muted/50">
          <p className="text-sm text-muted-foreground">{getRiskMessage()}</p>
        </div>

        <div className="text-xs text-muted-foreground border-t pt-4">
          <p className="font-semibold mb-1">Important Notice:</p>
          <p>
            This assessment is for informational purposes only and should not be considered a
            medical diagnosis. Please consult with a qualified healthcare professional for proper
            medical advice and diagnosis.
          </p>
        </div>
      </CardContent>
    </Card>
  );
};
