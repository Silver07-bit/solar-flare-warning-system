import { useEffect, useState } from "react";
import { cn } from "../../lib/utils";

export const TextGenerateEffect = ({
  words,
  className,
}: {
  words: string;
  className?: string;
}) => {
  const [displayedText, setDisplayedText] = useState("");

  useEffect(() => {
    let index = 0;
    setDisplayedText("");
    const interval = setInterval(() => {
      if (index < words.length) {
        setDisplayedText((prev) => prev + words.charAt(index));
        index++;
      } else {
        clearInterval(interval);
      }
    }, 8);

    return () => clearInterval(interval);
  }, [words]);

  return (
    <div className={cn("font-mono text-xs text-slate-300 leading-relaxed whitespace-pre-wrap", className)}>
      {displayedText}
      <span className="inline-block w-2 h-4 ml-1 bg-cyan-400 animate-pulse align-middle" />
    </div>
  );
};