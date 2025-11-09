import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { generateSamplePlan, HealthPlan, PlanType, PLAN_THRESHOLD } from "@/services/api";
const sanitizeMarkdown = (input?: string) =>
  input?.replace(/[#*_`>/\-]/g, "").replace(/\n+/g, "\n");
import { toast } from "sonner";

const PLAN_TYPE_OPTIONS: { label: string; value: PlanType }[] = [
  { label: "Workout", value: "workout" },
  { label: "Diet", value: "diet" },
  { label: "Workout + Diet", value: "both" },
];

export const PlanGeneratorCard = () => {
  const [planType, setPlanType] = useState<PlanType>("both");
  const [probability, setProbability] = useState("0.5");
  const [plan, setPlan] = useState<HealthPlan | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleGeneratePlan = async () => {
    const parsedProbability = parseFloat(probability);
    const probabilityValue = Number.isFinite(parsedProbability) ? parsedProbability : undefined;

    try {
      setIsLoading(true);
      const result = await generateSamplePlan(planType, probabilityValue);
      setPlan(result);
    } catch (error) {
      console.error("Sample plan generation failed", error);
      toast.error("Unable to generate plan. Please confirm the backend is running and try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const renderPlan = () => {
    if (!plan) return null;

    const sections: JSX.Element[] = [];
    const sanitizedWorkout = plan.workoutPlan?.trim();
    const sanitizedDiet = plan.dietPlan?.trim();

    if (sanitizedWorkout || plan.planType === "workout" || plan.planType === "both") {
      sections.push(
        <div key="workout">
          <p className="font-semibold mb-1">Workout Plan</p>
          <Textarea
            value={sanitizeMarkdown(sanitizedWorkout) || plan.workoutPlan || "No workout guidance returned. Try regenerating."}
            readOnly
            className="min-h-[160px]"
          />
        </div>
      );
    }

    if (sanitizedDiet || plan.planType === "diet" || plan.planType === "both") {
      sections.push(
        <div key="diet">
          <p className="font-semibold mb-1">Nutrition Plan</p>
          <Textarea
            value={sanitizeMarkdown(sanitizedDiet) || plan.dietPlan || "No nutrition guidance returned. Try regenerating."}
            readOnly
            className="min-h-[160px]"
          />
        </div>
      );
    }

    if (!sections.length) {
      sections.push(
        <div key="fallback" className="space-y-2">
          <p className="font-semibold">Plan Details</p>
          <Textarea
            value={JSON.stringify(plan, null, 2)}
            readOnly
            className="min-h-[160px] font-mono text-xs"
          />
        </div>
      );
    }

    return <div className="space-y-4">{sections}</div>;
  };

  return (
    <Card className="shadow-md">
      <CardHeader>
        <CardTitle>Instant Wellness Plan</CardTitle>
        <CardDescription>
          Skip the questionnaire and ask Gemini to craft a workout or diet plan using sample data.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Plan Type</label>
            <Select
              value={planType}
              onValueChange={(value) => setPlanType(value as PlanType)}
            >
              <SelectTrigger>
                <SelectValue placeholder="Choose plan type" />
              </SelectTrigger>
              <SelectContent>
                {PLAN_TYPE_OPTIONS.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium">Probability (0-1)</label>
            <Input
              type="number"
              min={PLAN_THRESHOLD}
              step="0.05"
              value={probability}
              onChange={(event) => setProbability(event.target.value)}
              placeholder="0.5"
            />
          </div>
        </div>

        <Button onClick={handleGeneratePlan} disabled={isLoading} className="w-full">
          {isLoading ? "Generating..." : "Generate Plan"}
        </Button>

        {renderPlan()}
      </CardContent>
    </Card>
  );
};
