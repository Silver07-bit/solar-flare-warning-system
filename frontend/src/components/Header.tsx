import React, { useState, useEffect } from "react";
import { AlertTriangle, ShieldCheck, Sun, Compass } from "lucide-react";
import { MovingBorderBadge } from "./ui/moving-border";

interface HeaderProps {
  scenario: string;
  setScenario: (sc: string) => void;
  riskLevel: string;
  targetAR: string;
}

export const Header: React.FC<HeaderProps> = ({
  scenario,
  setScenario,
  riskLevel,
  targetAR,
}) => {
  const [utcTime, setUtcTime] = useState("");
  const [istTime, setIstTime] = useState("");

  useEffect(() => {
    const updateClocks = () => {
      const now = new Date();
      setUtcTime(now.toUTCString().slice(17, 25) + " UTC");
      setIstTime(
        now.toLocaleTimeString("en-IN", {
          timeZone: "Asia/Kolkata",
          hour12: false,
        }) + " IST"
      );
    };
    updateClocks();
    const interval = setInterval(updateClocks, 1000);
    return () => clearInterval(interval);
  }, []);

  const badgeColor =
    riskLevel === "CRITICAL"
      ? "#ff334b"
      : riskLevel === "HIGH"
      ? "#ffb300"
      : "#00e676";

  return (
    <header className="relative z-20 border-b border-white/10 bg-space-950/80 backdrop-blur-xl px-6 py-4">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        {/* Mission Title */}
        <div className="flex items-center gap-3.5">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20 border border-cyan-300/30">
            <Sun className="h-5 w-5 text-white animate-spin-slow" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold px-2 py-0.5 rounded bg-orange-500/20 text-orange-400 border border-orange-500/30">
                ISRO ADITYA-L1
              </span>
              <span className="text-xs text-slate-400 font-mono">
                SUIT PAYLOAD (279.6 nm)
              </span>
            </div>
            <h1 className="text-xl font-extrabold tracking-tight text-white flex items-center gap-2">
              Solar Flare & Space Weather Early Warning System
              <span className="text-xs font-mono text-cyan-400 font-normal">
                [{targetAR}]
              </span>
            </h1>
          </div>
        </div>

        {/* Live Controls & Telemetry */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Scenario Picker */}
          <div className="flex items-center gap-2 bg-space-900 border border-white/10 rounded-xl px-3 py-1.5 text-xs">
            <Compass className="h-4 w-4 text-cyan-400" />
            <select
              value={scenario}
              onChange={(e) => setScenario(e.target.value)}
              className="bg-transparent text-slate-200 outline-none cursor-pointer text-xs font-medium"
            >
              <option value="AR3664_Impending_X_Flare" className="bg-space-900">
                AR-13664 (Superflare Eruption)
              </option>
              <option value="AR3685_M_Class_Eruption" className="bg-space-900">
                AR-12673 (Sept 2017 X9.3 Flare)
              </option>
              <option value="AR3670_Quiet_Sun" className="bg-space-900">
                AR-13100 (Quiet Solar Baseline)
              </option>
            </select>
          </div>

          {/* DEFCON Warning Badge */}
          <MovingBorderBadge color={badgeColor}>
            {riskLevel === "CRITICAL" ? (
              <AlertTriangle className="h-3.5 w-3.5 text-crimson-glow" />
            ) : (
              <ShieldCheck className="h-3.5 w-3.5 text-cyan-glow" />
            )}
            <span>DEFCON {riskLevel}</span>
          </MovingBorderBadge>

          {/* Clocks */}
          <div className="bg-space-900/90 border border-white/10 rounded-xl px-3.5 py-1.5 font-mono text-xs text-slate-300 flex items-center gap-3 shadow-inner">
            <div className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-emerald-400 animate-ping" />
              <span>{utcTime}</span>
            </div>
            <span className="text-white/20">|</span>
            <span className="text-slate-400">{istTime}</span>
          </div>
        </div>
      </div>
    </header>
  );
};
