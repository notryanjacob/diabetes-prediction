import { cn } from "@/lib/utils";
import { Bot, User } from "lucide-react";

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  isLoading?: boolean;
}

export const ChatMessage = ({ role, content, isLoading }: ChatMessageProps) => {
  const isAssistant = role === "assistant";

  return (
    <div
      className={cn(
        "flex gap-3 p-4 rounded-lg animate-slide-up",
        isAssistant ? "bg-card" : "bg-primary/5"
      )}
    >
      <div
        className={cn(
          "flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
          isAssistant
            ? "bg-gradient-to-br from-primary to-primary-glow text-primary-foreground"
            : "bg-secondary text-secondary-foreground"
        )}
      >
        {isAssistant ? <Bot className="h-5 w-5" /> : <User className="h-5 w-5" />}
      </div>
      <div className="flex-1 space-y-2">
        <p className="text-sm font-medium">
          {isAssistant ? "AI Assistant" : "You"}
        </p>
        {isLoading ? (
          <div className="flex gap-1">
            <span className="h-2 w-2 bg-primary rounded-full animate-pulse-subtle" />
            <span className="h-2 w-2 bg-primary rounded-full animate-pulse-subtle" style={{ animationDelay: "0.2s" }} />
            <span className="h-2 w-2 bg-primary rounded-full animate-pulse-subtle" style={{ animationDelay: "0.4s" }} />
          </div>
        ) : (
          <p className="text-sm text-foreground/90 whitespace-pre-wrap">{content}</p>
        )}
      </div>
    </div>
  );
};
