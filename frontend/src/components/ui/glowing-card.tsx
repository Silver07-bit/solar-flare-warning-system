import React from "react";
import { cn } from "../../lib/utils";

export const GlowingCard = ({
  children,
  className,
  glowColor = "rgba(0, 229, 255, 0.25)",
}: {
  children: React.ReactNode;
  className?: string;
  glowColor?: string;
}) => {
  return (
    <div
      className={cn(
        "relative group/card rounded-2xl border border-white/10 bg-space-900/85 p-6 backdrop-blur-xl transition-all duration-300 hover:border-cyan-400/35 hover:shadow-2xl",
        className
      )}
      style={{
        boxShadow: `0 0 20px -6px ${glowColor}`,
      }}
    >
      <div className="absolute -inset-px rounded-2xl bg-gradient-to-r from-cyan-500/15 via-blue-500/10 to-cyan-400/15 opacity-0 blur-sm transition-opacity duration-500 group-hover/card:opacity-100 pointer-events-none" />
      <div className="relative z-10">{children}</div>
    </div>
  );
};

