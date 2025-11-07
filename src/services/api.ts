/**
 * API Service for connecting to FastAPI backend
 * 
 * This file contains boilerplate code to connect the frontend to your FastAPI backend.
 * Replace the BASE_URL with your actual FastAPI server URL.
 */

// TODO: Replace this with your actual FastAPI backend URL
const BASE_URL = "http://localhost:8000"; // Change this to your FastAPI server URL

export interface DiabetesFeatures {
  heavyAlcoholConsumption: boolean;
  difficultyWalking: boolean;
  // Add more features as needed based on your ML model
  // Example additional features:
  // age?: number;
  // bmi?: number;
  // highBloodPressure?: boolean;
  // highCholesterol?: boolean;
  // smoker?: boolean;
  // physicalActivity?: boolean;
  // fruits?: boolean;
  // veggies?: boolean;
  // [key: string]: any;
}

export interface PredictionResponse {
  probability: number;
  riskLevel: "low" | "moderate" | "high";
  message?: string;
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
    
    // Transform the response from your ML model to match our interface
    // Adjust this based on your actual API response format
    return {
      probability: data.probability || data.score || 0,
      riskLevel: getRiskLevel(data.probability || data.score || 0),
      message: data.message,
    };
  } catch (error) {
    console.error("Error calling FastAPI backend:", error);
    throw error;
  }
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
