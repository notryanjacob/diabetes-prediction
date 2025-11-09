import { useEffect, useRef } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ChatMessage } from "@/components/ChatMessage";
import { ChatInput } from "@/components/ChatInput";
import { ResultsCard } from "@/components/ResultsCard";
import { Button } from "@/components/ui/button";
import { useDiabetesChat } from "@/hooks/useDiabetesChat";
import { Activity, RefreshCw } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { PlanGeneratorCard } from "@/components/PlanGeneratorCard";

const Index = () => {
  const { messages, isLoading, results, awaitingPlanChoice, sendMessage, resetChat } = useDiabetesChat();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);
  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-primary/5">
      {/* Header */}
      <header className="border-b bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-full bg-gradient-to-br from-primary to-primary-glow flex items-center justify-center">
                <Activity className="h-6 w-6 text-primary-foreground" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                  Diabetes Prediction using Agentic AI
                </h1>
                <p className="text-sm text-muted-foreground">Powered by Machine Learning & Gemini AI</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8 max-w-4xl">
        <div className="space-y-6">
          {/* Hero Section */}
          <div className="text-center space-y-2 animate-slide-up">
            <h2 className="text-3xl font-bold">AI-Powered Health Assessment</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Our advanced machine learning model, combined with conversational AI, helps assess
              your diabetes risk through a simple, interactive questionnaire.
            </p>
          </div>

          {/* Chat Interface */}
          <Card className="shadow-lg">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Interactive Assessment</CardTitle>
                  <CardDescription>Answer the questions to receive your risk assessment</CardDescription>
                </div>
                {results && (
                  <Button
                    onClick={resetChat}
                    variant="outline"
                    size="sm"
                    className="gap-2"
                  >
                    <RefreshCw className="h-4 w-4" />
                    Start Over
                  </Button>
                )}
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <ScrollArea className="h-[400px] pr-4">
                  <div className="space-y-4">
                    {messages.map((message, index) => (
                      <ChatMessage
                        key={index}
                        role={message.role}
                        content={message.content}
                      />
                    ))}
                    {isLoading && (
                      <ChatMessage role="assistant" content="" isLoading />
                    )}
                    <div ref={messagesEndRef} />
                  </div>
                </ScrollArea>

                {(!results || awaitingPlanChoice) && (
                  <div className="pt-4 border-t">
                    <ChatInput
                      onSend={sendMessage}
                      disabled={isLoading}
                      placeholder={
                        awaitingPlanChoice
                          ? "Type workout / diet / both / no..."
                          : "Type your answer..."
                      }
                    />
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Results */}
          {results && (
            <ResultsCard
              probability={results.probability}
              riskLevel={results.riskLevel}
            />
          )}

          {/* Info Cards */}
          <div className="grid md:grid-cols-3 gap-4 animate-fade-in">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">AI-Powered</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Uses Gemini 2.5 Flash for natural conversation flow and advanced ML model for accurate predictions.
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Quick Assessment</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Get your diabetes risk assessment in minutes through simple yes/no and value-based questions.
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Evidence-Based</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Our model is trained on medical data to provide reliable risk assessments.
                </p>
              </CardContent>
            </Card>
          </div>

          <PlanGeneratorCard />
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t mt-12 py-6">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          <p>
            This tool is for informational purposes only and does not constitute medical advice.
            Always consult a healthcare professional for medical concerns.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
