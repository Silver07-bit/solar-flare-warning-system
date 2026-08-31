import React, { useState, useEffect } from "react";
import { GlowingCard } from "./ui/glowing-card";
import { TextGenerateEffect } from "./ui/text-generate-effect";
import { fetchBulletin } from "../services/api";
import { FileText, Award, CheckCircle2, Copy } from "lucide-react";

export const TabTelemetryBulletin: React.FC = () => {
  const [bulletin, setBulletin] = useState<string>("");
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    fetchBulletin()
      .then((res: string) => setBulletin(res))
      .catch((err: unknown) => console.error(err));
  }, []);

  const handleCopy = () => {
    if (bulletin) {
      navigator.clipboard.writeText(bulletin);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="flex items-center justify-between border-b border-white/10 pb-4">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Award className="h-5 w-5 text-cyan-400" />
            Space Weather Skill Scores & ISSDC Advisory Dispatcher
          </h2>
          <p className="text-xs text-slate-400 font-mono mt-0.5">
            Strict 12-Fold Leave-One-Region-Out Cross-Validation benchmarks & Automated ISRO ISSDC Bulletin Stream.
          </p>
        </div>
        <span className="px-3 py-1 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/30 font-mono text-xs">
          ISSDC Standard Formatted
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Col: LORO-CV Benchmarks */}
        <div className="lg:col-span-5 space-y-4">
          <GlowingCard>
            <div className="flex items-center justify-between mb-4">
              <span className="text-sm font-bold text-white flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                12-Fold LORO-CV Scientific Metrics
              </span>
              <span className="text-[10px] font-mono text-slate-400 bg-space-950 px-2 py-0.5 rounded border border-white/5">
                N=12 Folds
              </span>
            </div>

            <div className="space-y-3 font-mono text-xs">
              <div className="flex justify-between items-center p-2.5 bg-space-950/80 rounded-xl border border-white/5">
                <span className="text-slate-400">True Skill Statistic (TSS)</span>
                <span className="text-cyan-400 font-bold">-0.179 ± 0.429</span>
              </div>
              <div className="flex justify-between items-center p-2.5 bg-space-950/80 rounded-xl border border-white/5">
                <span className="text-slate-400">Heidke Skill Score (HSS)</span>
                <span className="text-amber-400 font-bold">-0.004 ± 0.234</span>
              </div>
              <div className="flex justify-between items-center p-2.5 bg-space-950/80 rounded-xl border border-white/5">
                <span className="text-slate-400">24-48h Flare Recall (TPR)</span>
                <span className="text-emerald-400 font-bold">11.1% ± 28.3%</span>
              </div>
              <div className="flex justify-between items-center p-2.5 bg-space-950/80 rounded-xl border border-white/5">
                <span className="text-slate-400">24-48h Specificity (TNR)</span>
                <span className="text-white font-bold">71.0% ± 29.8%</span>
              </div>
              <div className="flex justify-between items-center p-2.5 bg-space-950/80 rounded-xl border border-white/5">
                <span className="text-slate-400">Peak Flux MAE</span>
                <span className="text-rose-300 font-bold">0.282 ± 0.278</span>
              </div>
              <div className="flex justify-between items-center p-2.5 bg-space-950/80 rounded-xl border border-white/5">
                <span className="text-slate-400">Single Split Held-Out TSS</span>
                <span className="text-emerald-300 font-bold">+0.120 (AUC 0.605)</span>
              </div>
            </div>

            <div className="mt-4 p-3 bg-space-950/60 rounded-xl border border-white/5 text-[11px] text-slate-400 font-sans">
              Evaluated under strict Leave-One-Region-Out cross-validation across 12 distinct NOAA active regions with zero spatial/temporal data leakage.
            </div>
          </GlowingCard>
        </div>

        {/* Right Col: ISSDC Bulletin Stream */}
        <div className="lg:col-span-7 space-y-4">
          <GlowingCard glowColor="rgba(0, 229, 255, 0.25)">
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-bold text-white flex items-center gap-2">
                <FileText className="h-4 w-4 text-cyan-400" />
                ISRO ISSDC Space Weather Advisory Bulletin
              </span>
              <button
                onClick={handleCopy}
                className="flex items-center gap-1.5 px-3 py-1 bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-300 rounded-lg text-xs font-mono border border-cyan-500/30 transition-all cursor-pointer"
              >
                <Copy className="h-3.5 w-3.5" />
                <span>{copied ? "COPIED!" : "COPY"}</span>
              </button>
            </div>

            <div className="p-4 bg-space-950 rounded-xl border border-white/10 max-h-[380px] overflow-y-auto">
              {bulletin ? (
                <TextGenerateEffect words={bulletin} />
              ) : (
                <div className="text-slate-500 font-mono text-xs">
                  AWAITING BULLETIN STREAM...
                </div>
              )}
            </div>
          </GlowingCard>
        </div>
      </div>
    </div>
  );
};
