import React from "react";
import { motion } from "framer-motion";
import { cn } from "../../lib/utils";

export function MovingBorderBadge({
  children,
  className,
  color = "#00e5ff",
}: {
  children: React.ReactNode;
  className?: string;
  color?: string;
}) {
  return (
    <div className={cn("relative p-[1.5px] overflow-hidden rounded-full inline-flex", className)}>
      <motion.div
        className="absolute inset-0"
        style={{
          background: `radial-gradient(circle at center, ${color} 20%, transparent 70%)`,
        }}
        animate={{
          rotate: [0, 360],
        }}
        transition={{
          duration: 3.5,
          repeat: Infinity,
          ease: "linear",
        }}
      />
      <div className="relative z-10 px-3.5 py-1 rounded-full bg-space-950/90 text-xs font-semibold backdrop-blur-md flex items-center gap-1.5 border border-white/10">
        {children}
      </div>
    </div>
  );
}