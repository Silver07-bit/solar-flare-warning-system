import React from "react";
import type { PredictResponse } from "../services/api";
import { ShieldAlert, Navigation, Zap, Plane, Users, Globe2 } from "lucide-react";

export const TabImpactMatrix: React.FC<{ prediction: PredictResponse | null }> = ({
  prediction,
}) => {
  if (!prediction) return null;

  const directives = prediction.mitigation_directives || [];

  const iconsMap: Record<string, React.ReactNode> = {
    "ISRO NavIC & Satellites": <Navigation className="h-5 w-5 text-cyan-400" />,
    "National Power Grid (PGCIL)": <Zap className="h-5 w-5 text-amber-400" />,
    "Aviation & Communication": <Plane className="h-5 w-5 text-blue-400" />,
    "Human Spaceflight (Gaganyaan)": <Users className="h-5 w-5 text-rose-400" />,
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between border-b border-white/10 pb-4">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Globe2 className="h-5 w-5 text-cyan-400" />
            Decision Support & National Asset Defense Matrix (SIH Focus)
          </h2>
          <p className="text-xs text-slate-400 font-mono mt-0.5">
            Automated translation of learned flare probabilities into operational defense protocols for Indian infrastructure.
          </p>
        </div>
        <span className="px-3 py-1 rounded-full bg-rose-500/10 text-rose-300 border border-rose-500/30 font-mono text-xs">
          NOAA R/G/S Scale Integrated
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {directives.map((d: { sector: string; status: string; directive: string; level: string }, idx: number) => {
          const isRed = d.level === "RED";
          const isAmber = d.level === "AMBER";
          const borderColor = isRed
            ? "border-rose-500/40 bg-rose-950/10"
            : isAmber
            ? "border-amber-500/40 bg-amber-950/10"
            : "border-emerald-500/40 bg-emerald-950/10";
          const badgeColor = isRed
            ? "bg-rose-500/20 text-rose-300 border-rose-500/30"
            : isAmber
            ? "bg-amber-500/20 text-amber-300 border-amber-500/30"
            : "bg-emerald-500/20 text-emerald-300 border-emerald-500/30";

          return (
            <div
              key={idx}
              className={`rounded-2xl border p-5 backdrop-blur-xl transition-all duration-300 hover:shadow-2xl ${borderColor}`}
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2.5">
                  {iconsMap[d.sector] || <ShieldAlert className="h-5 w-5 text-cyan-400" />}
                  <span className="font-bold text-white text-sm">{d.sector}</span>
                </div>
                <span className={`px-2.5 py-0.5 rounded-full text-xs font-bold font-mono border ${badgeColor}`}>
                  {d.status}
                </span>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed font-sans mt-2">
                {d.directive}
              </p>
            </div>
          );
        })}
      </div>

      <div className="rounded-2xl border border-white/10 bg-space-950/80 p-5 font-mono text-xs text-slate-300 space-y-2 shadow-xl">
        <span className="text-cyan-400 font-bold block text-sm">
          🇮🇳 STRATEGIC DEFENCE PROTOCOL COMPLIANCE
        </span>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
          <div>
            <span className="text-slate-400 block text-[10px]">NAVIC MITIGATION</span>
            <span className="text-white">Differential Ionospheric Delay broadcast active</span>
          </div>
          <div>
            <span className="text-slate-400 block text-[10px]">PGCIL 765KV CORRIDOR</span>
            <span className="text-white">Transformer neutral DC blocking pre-armed</span>
          </div>
          <div>
            <span className="text-slate-400 block text-[10px]">GAGANYAAN CREW DOSAGE</span>
            <span className="text-white">LEO orbital EVA restriction window flagged</span>
          </div>
        </div>
      </div>
    </div>
  );
};