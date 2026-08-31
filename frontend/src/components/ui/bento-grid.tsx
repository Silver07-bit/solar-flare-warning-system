import React from "react";
import { cn } from "../../lib/utils";

export const BentoGrid = ({
  className,
  children,
}: {
  className?: string;
  children?: React.ReactNode;
}) => {
  return (
    <div
      className={cn(
        "grid md:auto-rows-[16rem] grid-cols-1 md:grid-cols-3 gap-4 max-w-7xl mx-auto",
        className
      )}
    >
      {children}
    </div>
  );
};

export const BentoGridItem = ({
  className,
  title,
  description,
  header,
  icon,
}: {
  className?: string;
  title?: string | React.ReactNode;
  description?: string | React.ReactNode;
  header?: React.ReactNode;
  icon?: React.ReactNode;
}) => {
  return (
    <div
      className={cn(
        "row-span-1 rounded-2xl group/bento hover:shadow-2xl transition duration-300 shadow-input border border-white/10 bg-space-900/90 p-5 backdrop-blur-md justify-between flex flex-col space-y-3 hover:border-cyan-glow/50 hover:bg-space-850",
        className
      )}
    >
      {header}
      <div className="group-hover/bento:translate-x-1 transition duration-200">
        <div className="flex items-center gap-2 mb-1.5">
          {icon}
          <div className="font-sans font-bold text-white text-sm">
            {title}
          </div>
        </div>
        <div className="font-sans font-normal text-slate-300 text-xs leading-relaxed">
          {description}
        </div>
      </div>
    </div>
  );
};
