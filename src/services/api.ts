/**
 * API Service for connecting to FastAPI backend
 * 
 * This file contains boilerplate code to connect the frontend to your FastAPI backend.
 * Replace the BASE_URL with your actual FastAPI server URL.
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const MODEL_THRESHOLD = 0.674;
export const PLAN_THRESHOLD = 0.4;

export interface DiabetesFeatures {
  HighBP: boolean;
  HighChol: boolean;
  BMI: number;
  Smoker: boolean;
  Stroke: boolean;
  HeartDiseaseorAttack: boolean;
  CholCheck: boolean;
  PhysActivity: boolean;
  Fruits: boolean;
  Veggies: boolean;
  HvyAlcoholConsump: boolean;
  GenHlth: number;
  Age: number;
  DiffWalk: boolean;
  PhysHlth: number;
  MentHlth: number;
  NoDocbcCost: boolean;
  Education: number;
  Income: number;
  Sex: boolean; // true = male, false = female
  AnyHealthcare: boolean;
}

export type PlanType = "workout" | "diet" | "both";

export interface PredictionResponse {
  probability: number;
  riskLevel: "low" | "moderate" | "high";
  diabetesScore: number;
  predictedLabel: 0 | 1;
  message?: string;
  plan?: HealthPlan;
}

export interface HealthPlan {
  planType: PlanType;
  workoutPlan?: string;
  dietPlan?: string;
  prompt: string;
}

export interface PlanRequest {
  features: DiabetesFeatures;
  probability: number;
  planType: PlanType;
}

/**
 * Send features to FastAPI backend for diabetes prediction
 * 
 * @param features - Object containing all the features collected from the user
 * @returns Prediction response with probability and risk level
 */
export async function predictDiabetes(features: DiabetesFeatures): Promise<PredictionResponse> {
  try {
    const response = await fetch(`${BASE_URL}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(features),
    });

    if (!response.ok) {
      throw new Error(`API request failed with status ${response.status}`);
    }

    const data = await response.json();
    const probability = typeof data.probability === "number" ? data.probability : 0;
    const riskLevel = data.riskLevel ?? getRiskLevel(probability);
    const diabetesScore = data.diabetesScore ?? Number((probability * 10).toFixed(1));
    const predictedLabel =
      typeof data.predictedLabel === "number"
        ? (data.predictedLabel as 0 | 1)
        : (probability >= MODEL_THRESHOLD ? 1 : 0);

    return {
      probability,
      riskLevel,
      diabetesScore,
      predictedLabel,
      message: data.message,
      plan: data.plan,
    };
  } catch (error) {
    console.error("Error calling FastAPI backend:", error);
    throw error;
  }
}

export async function generatePlan(payload: PlanRequest): Promise<HealthPlan> {
  if (payload.probability < PLAN_THRESHOLD) {
    throw new Error("Plan generation is only available for higher risk cases.");
  }

  const response = await fetch(`${BASE_URL}/plan`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`Plan generation failed with status ${response.status}`);
  }

  return response.json();
}

export async function generateSamplePlan(
  planType: PlanType = "both",
  probability?: number
): Promise<HealthPlan> {
  const url = new URL(`${BASE_URL}/plan/sample`);
  url.searchParams.set("plan_type", planType);
  if (typeof probability === "number" && Number.isFinite(probability)) {
    url.searchParams.set("probability", probability.toString());
  }

  const response = await fetch(url.toString(), {
    method: "POST",
  });

  if (!response.ok) {
    throw new Error(`Sample plan generation failed with status ${response.status}`);
  }

  return response.json();
}

/**
 * Determine risk level based on probability
 */
function getRiskLevel(probability: number): "low" | "moderate" | "high" {
  if (probability < 0.3) return "low";
  if (probability < 0.6) return "moderate";
  return "high";
}

/**
 * Health check endpoint to verify backend connectivity
 */
export async function healthCheck(): Promise<boolean> {
  try {
    const response = await fetch(`${BASE_URL}/health`);
    return response.ok;
  } catch (error) {
    console.error("Backend health check failed:", error);
    return false;
  }
}
