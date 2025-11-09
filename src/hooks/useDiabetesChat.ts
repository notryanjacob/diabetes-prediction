import { useState } from "react";
import {
  DiabetesFeatures,
  predictDiabetes,
  PredictionResponse,
  generatePlan,
  HealthPlan,
  PlanType,
} from "@/services/api";
import { toast } from "sonner";

interface Message {
  role: "user" | "assistant";
  content: string;
}

interface ChatState {
  messages: Message[];
  isLoading: boolean;
  features: Partial<DiabetesFeatures>;
  currentQuestion: number;
  results: PredictionResponse | null;
  awaitingPlanChoice: boolean;
}

type QuestionType = "boolean" | "number" | "integer" | "choice";

const PLAN_THRESHOLD = 0.4;

interface Question<T extends keyof DiabetesFeatures = keyof DiabetesFeatures> {
  key: T;
  question: string;
  type: QuestionType;
  min?: number;
  max?: number;
  choices?: {
    label: string;
    value: DiabetesFeatures[T];
    synonyms: string[];
  }[];
}

const QUESTIONS: Question[] = [
  {
    key: "HighBP",
    question: "Have you ever been told by a health professional that you have high blood pressure? (yes/no)",
    type: "boolean",
  },
  {
    key: "HighChol",
    question: "Have you ever been told you have high cholesterol? (yes/no)",
    type: "boolean",
  },
  {
    key: "BMI",
    question: "What is your current Body Mass Index (BMI)? Please provide a number between 10 and 60.",
    type: "number",
    min: 10,
    max: 60,
  },
  {
    key: "Smoker",
    question: "Have you smoked at least 100 cigarettes in your lifetime? (yes/no)",
    type: "boolean",
  },
  {
    key: "Stroke",
    question: "Have you ever been told by a doctor that you had a stroke? (yes/no)",
    type: "boolean",
  },
  {
    key: "HeartDiseaseorAttack",
    question: "Have you ever had coronary heart disease or a heart attack? (yes/no)",
    type: "boolean",
  },
  {
    key: "CholCheck",
    question: "Have you had your cholesterol checked within the past 5 years? (yes/no)",
    type: "boolean",
  },
  {
    key: "PhysActivity",
    question: "In the past 30 days, did you participate in any physical activity or exercise outside of regular work? (yes/no)",
    type: "boolean",
  },
  {
    key: "Fruits",
    question: "Do you eat fruit at least once per day? (yes/no)",
    type: "boolean",
  },
  {
    key: "Veggies",
    question: "Do you eat vegetables at least once per day? (yes/no)",
    type: "boolean",
  },
  {
    key: "HvyAlcoholConsump",
    question: "Would you describe your alcohol consumption as heavy (men >14 drinks/week, women >7)? (yes/no)",
    type: "boolean",
  },
  {
    key: "GenHlth",
    question: "How would you rate your general health on a scale of 1 (excellent) to 5 (poor)?",
    type: "integer",
    min: 1,
    max: 5,
  },
  {
    key: "Age",
    question: "What is your age in years? (18-120)",
    type: "integer",
    min: 18,
    max: 120,
  },
  {
    key: "DiffWalk",
    question: "Do you have serious difficulty walking or climbing stairs? (yes/no)",
    type: "boolean",
  },
  {
    key: "PhysHlth",
    question: "During the past 30 days, on how many days was your physical health not good? (0-30)",
    type: "integer",
    min: 0,
    max: 30,
  },
  {
    key: "MentHlth",
    question: "During the past 30 days, on how many days was your mental health not good? (0-30)",
    type: "integer",
    min: 0,
    max: 30,
  },
  {
    key: "NoDocbcCost",
    question: "In the past 12 months, was there a time you needed to see a doctor but could not because of cost? (yes/no)",
    type: "boolean",
  },
  {
    key: "Education",
    question:
      "What is the highest level of education you completed? (1=Never attended, 2=Grades 1-8, 3=Grades 9-11, 4=Grade 12 or GED, 5=Some college/technical school, 6=College graduate)",
    type: "integer",
    min: 1,
    max: 6,
  },
  {
    key: "Income",
    question:
      "What is your annual household income category? (1=<$10k, 2=$10-15k, 3=$15-20k, 4=$20-25k, 5=$25-35k, 6=$35-50k, 7=$50-75k, 8=>=$75k)",
    type: "integer",
    min: 1,
    max: 8,
  },
  {
    key: "Sex",
    question: "What is your sex assigned at birth? (male/female)",
    type: "choice",
    choices: [
      { label: "Male", value: true, synonyms: ["male", "m", "man"] },
      { label: "Female", value: false, synonyms: ["female", "f", "woman"] },
    ],
  },
  {
    key: "AnyHealthcare",
    question: "Do you currently have any kind of health coverage (insurance, HMO, or government plan)? (yes/no)",
    type: "boolean",
  },
];

const INTRO_MESSAGE = `Hello! I'm your AI health assistant. I'll ask you a series of quick questions to estimate your diabetes risk. Please answer with yes/no or the requested number.\n\n${QUESTIONS[0]?.question ?? ""}`;

const buildInitialState = (): ChatState => ({
  messages: [
    {
      role: "assistant",
      content: INTRO_MESSAGE,
    },
  ],
  isLoading: false,
  features: {},
  currentQuestion: 0,
  results: null,
  awaitingPlanChoice: false,
});

const normalizeText = (value: string) => value.trim().toLowerCase();

const parseBooleanResponse = (raw: string): boolean => {
  const normalized = normalizeText(raw);
  if (["yes", "y", "true", "1"].includes(normalized)) {
    return true;
  }
  if (["no", "n", "false", "0"].includes(normalized)) {
    return false;
  }
  throw new Error("Please answer with yes or no.");
};

const parseNumericResponse = (raw: string, question: Question): number => {
  const value = question.type === "integer" ? parseInt(raw, 10) : parseFloat(raw);
  if (Number.isNaN(value)) {
    throw new Error("Please provide a numeric value.");
  }
  if (question.min !== undefined && value < question.min) {
    throw new Error(`Please provide a value between ${question.min} and ${question.max ?? "the requested range"}.`);
  }
  if (question.max !== undefined && value > question.max) {
    throw new Error(`Please provide a value between ${question.min ?? "the requested range"} and ${question.max}.`);
  }
  if (question.type === "integer" && !Number.isInteger(value)) {
    throw new Error("Please provide a whole number.");
  }
  return value;
};

const parseChoiceResponse = (
  raw: string,
  question: Question
): DiabetesFeatures[keyof DiabetesFeatures] => {
  if (!question.choices?.length) {
    throw new Error("Please respond with one of the listed options.");
  }
  const normalized = normalizeText(raw);
  for (const option of question.choices) {
    if (option.synonyms.some((syn) => normalizeText(syn) === normalized)) {
      return option.value;
    }
  }
  const labels = question.choices.map((option) => option.label).join(", ");
  throw new Error(`Please respond with one of the following options: ${labels}.`);
};

const parseUserResponse = (
  response: string,
  question: Question
): DiabetesFeatures[typeof question.key] => {
  switch (question.type) {
    case "boolean":
      return parseBooleanResponse(response) as DiabetesFeatures[typeof question.key];
    case "number":
    case "integer":
      return parseNumericResponse(response, question) as DiabetesFeatures[typeof question.key];
    case "choice":
      return parseChoiceResponse(response, question) as DiabetesFeatures[typeof question.key];
    default:
      throw new Error("Unsupported question type.");
  }
};

const formatPlanMessage = (plan: HealthPlan) => {
  const headerType =
    plan.planType === "both"
      ? "comprehensive workout and nutrition plan"
      : `${plan.planType} plan`;

  const sections: string[] = [`Here is your personalized ${headerType}:`];

  if (plan.workoutPlan?.trim()) {
    sections.push(`**Workout Plan**\n${plan.workoutPlan.trim()}`);
  }

  if (plan.dietPlan?.trim()) {
    sections.push(`**Nutrition Plan**\n${plan.dietPlan.trim()}`);
  }

  if (sections.length === 1) {
    sections.push(
      "I wasn't able to include detailed steps this time, but please consult with a healthcare professional for personalized guidance."
    );
  }

  return sections.join("\n\n");
};

type PlanDecision = PlanType | "none";

const PLAN_PROMPT =
  "Would you like me to craft a personalized workout plan, diet plan, both, or would you prefer to skip this step? (Reply with “workout”, “diet”, “both”, or “no”.)";

const parsePlanChoice = (response: string): PlanDecision => {
  const normalized = normalizeText(response);
  if (!normalized) {
    throw new Error("Please let me know if you'd like a workout plan, diet plan, both, or no plan.");
  }

  if (["no", "nah", "skip", "later", "not now"].some((word) => normalized.includes(word))) {
    return "none";
  }

  const mentionsWorkout = /workout|exercise|fitness|training/.test(normalized);
  const mentionsDiet = /diet|meal|food|nutrition|plan/.test(normalized);
  const mentionsBoth = normalized.includes("both") || (mentionsWorkout && mentionsDiet);

  if (mentionsBoth) return "both";
  if (mentionsWorkout) return "workout";
  if (mentionsDiet) return "diet";

  throw new Error("Please reply with “workout”, “diet”, “both”, or “no”.");
};

export const useDiabetesChat = () => {
  const [state, setState] = useState<ChatState>(() => buildInitialState());

  const handlePlanChoice = async (userMessage: string, snapshot: ChatState) => {
    let decision: PlanDecision;

    try {
      decision = parsePlanChoice(userMessage);
    } catch (error) {
      const errorMessage =
        error instanceof Error
          ? error.message
          : "Please let me know if you'd like a workout plan, diet plan, both, or no plan.";

      setState((prev) => ({
        ...prev,
        messages: [
          ...prev.messages,
          {
            role: "assistant",
            content: `${errorMessage}\n\n${PLAN_PROMPT}`,
          },
        ],
        isLoading: false,
      }));
      return;
    }

    const prediction = snapshot.results;
    if (!prediction) {
      setState((prev) => ({
        ...prev,
        messages: [
          ...prev.messages,
          {
            role: "assistant",
            content:
              "I need to re-run the assessment before crafting a plan. Please restart the questionnaire.",
          },
        ],
        isLoading: false,
        awaitingPlanChoice: false,
      }));
      return;
    }

    if (decision === "none") {
      setState((prev) => ({
        ...prev,
        messages: [
          ...prev.messages,
          {
            role: "assistant",
            content:
              "No problem. If you change your mind later, just let me know and we can craft a plan anytime.",
          },
        ],
        isLoading: false,
        awaitingPlanChoice: false,
      }));
      return;
    }

    const completedFeatures = snapshot.features as DiabetesFeatures;
    const planType = decision as PlanType;

    try {
      const plan = await generatePlan({
        features: completedFeatures,
        probability: prediction.probability,
        planType,
      });

      const planMessage = formatPlanMessage(plan);

      setState((prev) => ({
        ...prev,
        messages: [
          ...prev.messages,
          {
            role: "assistant",
            content: planMessage,
          },
        ],
        isLoading: false,
        awaitingPlanChoice: false,
        results: prev.results ? { ...prev.results, plan } : prev.results,
      }));
    } catch (error) {
      console.error("Plan generation failed:", error);
      setState((prev) => ({
        ...prev,
        messages: [
          ...prev.messages,
          {
            role: "assistant",
            content: "I couldn't generate that plan just now. Would you like me to try again?",
          },
        ],
        isLoading: false,
        awaitingPlanChoice: true,
      }));
    }
  };

  const sendMessage = async (userMessage: string) => {
    const snapshot = state;

    setState((prev) => ({
      ...prev,
      messages: [...prev.messages, { role: "user", content: userMessage }],
      isLoading: true,
    }));

    if (snapshot.awaitingPlanChoice && snapshot.results) {
      await handlePlanChoice(userMessage, snapshot);
      return;
    }

    if (snapshot.results && snapshot.currentQuestion >= QUESTIONS.length) {
      setState((prev) => ({
        ...prev,
        messages: [
          ...prev.messages,
          {
            role: "assistant",
            content: "We've completed this assessment. Please restart if you'd like to run it again.",
          },
        ],
        isLoading: false,
      }));
      return;
    }

    const currentQuestion = QUESTIONS[state.currentQuestion];

    if (!currentQuestion) {
      setState((prev) => ({
        ...prev,
        messages: [
          ...prev.messages,
          {
            role: "assistant",
            content: "I've already completed the assessment. Please restart the chat to run another evaluation.",
          },
        ],
        isLoading: false,
      }));
      return;
    }

    let parsedValue: DiabetesFeatures[keyof DiabetesFeatures];

    try {
      parsedValue = parseUserResponse(userMessage, currentQuestion);
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "I couldn't interpret that response.";
      setState((prev) => ({
        ...prev,
        messages: [
          ...prev.messages,
          {
            role: "assistant",
            content: `${errorMessage}\n\n${currentQuestion.question}`,
          },
        ],
        isLoading: false,
      }));
      return;
    }

    const updatedFeatures: Partial<DiabetesFeatures> = {
      ...state.features,
      [currentQuestion.key]: parsedValue,
    };

    const nextQuestionIndex = state.currentQuestion + 1;

    if (nextQuestionIndex < QUESTIONS.length) {
      const nextQuestion = QUESTIONS[nextQuestionIndex];
      setState((prev) => ({
        ...prev,
        messages: [
          ...prev.messages,
          {
            role: "assistant",
            content: `Thank you. ${nextQuestion.question}`,
          },
        ],
        isLoading: false,
        features: updatedFeatures,
        currentQuestion: nextQuestionIndex,
      }));
      return;
    }

    setState((prev) => ({
      ...prev,
      messages: [
        ...prev.messages,
        {
          role: "assistant",
          content:
            "Thank you for answering all the questions. Let me analyze your responses and provide a risk assessment...",
        },
      ],
      features: updatedFeatures,
      currentQuestion: QUESTIONS.length,
    }));

    try {
      const completedFeatures = updatedFeatures as DiabetesFeatures;
      const prediction = await predictDiabetes(completedFeatures);
      const shouldOfferPlan = prediction.probability >= PLAN_THRESHOLD;
      const planInviteMessage = shouldOfferPlan
        ? {
            role: "assistant" as const,
            content: PLAN_PROMPT,
          }
        : null;

      setState((prev) => ({
        ...prev,
        messages: [
          ...prev.messages,
          {
            role: "assistant",
            content: `Based on your responses, your estimated diabetes risk probability is ${(prediction.probability * 100).toFixed(
              1
            )}%. ${prediction.message || "Please see the detailed results below."}\n\nDiabetes Score: ${prediction.diabetesScore.toFixed(
              1
            )} / 10`,
          },
          ...(planInviteMessage ? [planInviteMessage] : []),
        ],
        isLoading: false,
        results: prediction,
        awaitingPlanChoice: shouldOfferPlan,
        features: completedFeatures,
      }));
    } catch (error) {
      console.error("Error processing message:", error);
      toast.error("Failed to process your response. Please try again.");

      setState((prev) => ({
        ...prev,
        messages: [
          ...prev.messages,
          {
            role: "assistant",
            content:
              "I encountered an issue while analyzing your responses. Please try again or restart the assessment.",
          },
        ],
        isLoading: false,
      }));
    }
  };

  const resetChat = () => {
    setState(buildInitialState());
  };

  return {
    messages: state.messages,
    isLoading: state.isLoading,
    results: state.results,
    awaitingPlanChoice: state.awaitingPlanChoice,
    sendMessage,
    resetChat,
  };
};
