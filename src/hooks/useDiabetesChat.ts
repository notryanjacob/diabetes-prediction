import { useState } from "react";
import { DiabetesFeatures, predictDiabetes, PredictionResponse } from "@/services/api";
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
}

// Questions to ask the user
const QUESTIONS = [
  {
    key: "heavyAlcoholConsumption",
    question: "Do you consider yourself to be a heavy alcohol consumer? (once every 2-3 days)",
    type: "boolean" as const,
  },
  {
    key: "difficultyWalking",
    question: "In the recent times have you had any difficulty in walking without any injury?",
    type: "boolean" as const,
  },
  // Add more questions here based on your ML model features
  // Example:
  // {
  //   key: "age",
  //   question: "What is your age?",
  //   type: "number" as const,
  // },
];

export const useDiabetesChat = () => {
  const [state, setState] = useState<ChatState>({
    messages: [
      {
        role: "assistant",
        content: "Hello! I'm your AI health assistant. I'll ask you a few questions to assess your diabetes risk. Let's begin.\n\n" + QUESTIONS[0].question,
      },
    ],
    isLoading: false,
    features: {},
    currentQuestion: 0,
    results: null,
  });

  const parseUserResponse = (response: string, type: "boolean" | "number"): any => {
    if (type === "boolean") {
      const normalized = response.toLowerCase().trim();
      return normalized === "yes" || normalized === "y" || normalized === "true";
    }
    if (type === "number") {
      return parseFloat(response);
    }
    return response;
  };

  const sendMessage = async (userMessage: string) => {
    // Add user message
    setState((prev) => ({
      ...prev,
      messages: [...prev.messages, { role: "user", content: userMessage }],
      isLoading: true,
    }));

    try {
      // Parse and store the answer
      const currentQ = QUESTIONS[state.currentQuestion];
      const parsedValue = parseUserResponse(userMessage, currentQ.type);
      
      const updatedFeatures = {
        ...state.features,
        [currentQ.key]: parsedValue,
      };

      // Check if we have more questions
      const nextQuestionIndex = state.currentQuestion + 1;
      
      if (nextQuestionIndex < QUESTIONS.length) {
        // Ask next question
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
      } else {
        // All questions answered, make prediction
        setState((prev) => ({
          ...prev,
          messages: [
            ...prev.messages,
            {
              role: "assistant",
              content: "Thank you for answering all the questions. Let me analyze your responses and provide a risk assessment...",
            },
          ],
          features: updatedFeatures,
        }));

        // Call the ML model API
        const prediction = await predictDiabetes(updatedFeatures as DiabetesFeatures);
        
        setState((prev) => ({
          ...prev,
          messages: [
            ...prev.messages,
            {
              role: "assistant",
              content: `Based on your responses, I've completed the analysis. Your diabetes risk probability is ${(prediction.probability * 100).toFixed(1)}%. ${prediction.message || "Please see the detailed results below."}`,
            },
          ],
          isLoading: false,
          results: prediction,
        }));
      }
    } catch (error) {
      console.error("Error processing message:", error);
      toast.error("Failed to process your response. Please try again.");
      
      setState((prev) => ({
        ...prev,
        messages: [
          ...prev.messages,
          {
            role: "assistant",
            content: "I apologize, but I encountered an error processing your response. Please try again or contact support if the issue persists.",
          },
        ],
        isLoading: false,
      }));
    }
  };

  const resetChat = () => {
    setState({
      messages: [
        {
          role: "assistant",
          content: "Hello! I'm your AI health assistant. I'll ask you a few questions to assess your diabetes risk. Let's begin.\n\n" + QUESTIONS[0].question,
        },
      ],
      isLoading: false,
      features: {},
      currentQuestion: 0,
      results: null,
    });
  };

  return {
    messages: state.messages,
    isLoading: state.isLoading,
    results: state.results,
    sendMessage,
    resetChat,
  };
};
