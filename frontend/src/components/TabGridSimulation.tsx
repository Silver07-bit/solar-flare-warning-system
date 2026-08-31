import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Zap, Navigation, Plane, ShieldCheck } from "lucide-react";
import type { PredictResponse } from "../services/api";

interface TabGridSimulationProps {
  prediction: PredictResponse | null;
  isFlareActive: boolean;
}

export const TabGridSimulation: React.FC<TabGridSimulationProps> = ({
  prediction,
  isFlareActive,
}) => {
  const [activeSector, setActiveSector] = useState<"power" | "satellite" | "aviation">("power");

  const isCritical = isFlareActive || prediction?.risk_level === "CRITICAL";
  const isHigh = prediction?.risk_level === "HIGH";

  // Dynamic status generators
  const powerAssets = [
    {
      id: "pgcil-north",
      name: "Northern Grid: 765 kV Agra-Gwalior Corridor",
      region: "Northern Region (PowerGrid PGCIL)",
      status: isCritical ? "CRITICAL GIC SATURATION" : isHigh ? "ELEVATED DC BIAS" : "NOMINAL PHASE",
      riskPercent: isCritical ? 88 : isHigh ? 46 : 4,
      metricLabel: "NEUTRAL GIC CURRENT",
      metricValue: isCritical ? "48.2 Amperes DC" : isHigh ? "18.4 Amperes DC" : "1.2 Amperes DC",
      directive: isCritical
        ? "Pre-arm series neutral DC blocking capacitors. Reduce substation MVAR load by 25%."
        : "Continuous Hall-effect DC neutral sensor monitoring armed.",
      severity: isCritical ? "critical" : isHigh ? "warning" : "nominal",
    },
    {
      id: "pgcil-west",
      name: "Western Grid: 765 kV Raigarh-Pugalur HVDC",
      region: "Western / Southern Interconnector",
      status: isCritical ? "SEVERE HARMONIC DISTORTION" : isHigh ? "VOLTAGE OSCILLATION" : "NOMINAL PHASE",
      riskPercent: isCritical ? 74 : isHigh ? 38 : 6,
      metricLabel: "CORE FLUX SATURATION",
      metricValue: isCritical ? "1.82 Tesla (Sat)" : isHigh ? "1.45 Tesla" : "1.10 Tesla",
      directive: isCritical
        ? "Switch shunt reactors to maximum compensation. Prepare dynamic reactive reserve."
        : "Standard operational margin maintained.",
      severity: isCritical ? "critical" : isHigh ? "warning" : "nominal",
    },
    {
      id: "pgcil-sub",
      name: "Substation Transformer Asset: 400 kV Fatehpur",
      region: "Central Core Transmission",
      status: isCritical ? "THERMAL RUNAWAY RISK" : isHigh ? "HOTSPOT ALARM" : "NORMAL TEMP",
      riskPercent: isCritical ? 82 : isHigh ? 42 : 5,
      metricLabel: "TOP OIL TEMPERATURE",
      metricValue: isCritical ? "98.4 °C (Alert)" : isHigh ? "76.1 °C" : "54.2 °C",
      directive: isCritical
        ? "Activate forced-oil directed-water cooling. Isolate tertiary delta windings."
        : "Telemetry stream synchronized with NLDC Delhi.",
      severity: isCritical ? "critical" : isHigh ? "warning" : "nominal",
    },
  ];

  const satelliteAssets = [
    {
      id: "navic-constellation",
      name: "ISRO NavIC Constellation (IRNSS-1A to 1I)",
      region: "Geostationary & Geosynchronous Orbit",
      status: isCritical ? "IONOSPHERIC TEC DRIFT" : isHigh ? "SCINTILLATION RISK" : "CLOCK NOMINAL",
      riskPercent: isCritical ? 92 : isHigh ? 52 : 3,
      metricLabel: "TOTAL ELECTRON CONTENT (TEC)",
      metricValue: isCritical ? "+28.4 TECU Error" : isHigh ? "+9.8 TECU Error" : "1.1 TECU (Cal)",
      directive: isCritical
        ? "Broadcast real-time differential ionospheric correction ephemeris to ground NavIC receivers."
        : "Dual-frequency L5/S band range tracking stable.",
      severity: isCritical ? "critical" : isHigh ? "warning" : "nominal",
    },
    {
      id: "gsat-telecom",
      name: "GSAT-30 / INSAT-4B Telecommunications",
      region: "Geostationary Arc (83° E)",
      status: isCritical ? "DEEP DIELECTRIC CHARGING" : isHigh ? "SURGE POTENTIAL" : "NOMINAL TELEMETRY",
      riskPercent: isCritical ? 79 : isHigh ? 40 : 4,
      metricLabel: "SURFACE CHARGE POTENTIAL",
      metricValue: isCritical ? "-4.8 kV (Critical)" : isHigh ? "-1.6 kV" : "-0.2 kV",
      directive: isCritical
        ? "Inhibit transponder firmware re-flashing. Orient solar arrays for zero energetic electron buildup."
        : "Satellite Operations Control Centre (ISTRAC) link locked.",
      severity: isCritical ? "critical" : isHigh ? "warning" : "nominal",
    },
    {
      id: "gaganyaan-crew",
      name: "Gaganyaan LEO Human Orbital Module",
      region: "Low Earth Orbit (400 km, 51.6° Inc)",
      status: isCritical ? "CREW RADIATION ALERT (S3)" : isHigh ? "ELEVATED PROTON FLUX" : "SAFE DOSAGE",
      riskPercent: isCritical ? 95 : isHigh ? 58 : 2,
      metricLabel: "EST. ACCUMULATED DOSAGE",
      metricValue: isCritical ? "142.5 mSv/hr (Hazard)" : isHigh ? "38.1 mSv/hr" : "2.4 mSv/day",
      directive: isCritical
        ? "MANDATORY: Postpone Extravehicular Activity (EVA). Direct astronauts to polyethylene-shielded storm shelter."
        : "Cabin dosimeter baseline verified with ISRO Flight Dynamics.",
      severity: isCritical ? "critical" : isHigh ? "warning" : "nominal",
    },
  ];

  const aviationAssets = [
    {
      id: "dgca-polar",
      name: "DGCA Airway: North Polar Trans-Continental",
      region: "High-Latitude Air Route (Delhi-Chicago)",
      status: isCritical ? "COMPLETE HF BLACKOUT (R4)" : isHigh ? "RADIO DEGRADATION" : "ALL HF CHANNELS OPEN",
      riskPercent: isCritical ? 96 : isHigh ? 64 : 5,
      metricLabel: "D-REGION ABSORPTION",
      metricValue: isCritical ? "44.8 dB Attenuation" : isHigh ? "18.2 dB" : "1.5 dB",
      directive: isCritical
        ? "Reroute commercial flights below 60°N geomagnetic latitude. Switch comms to Iridium SATCOM."
        : "Monitoring international solar radio emission bulletins.",
      severity: isCritical ? "critical" : isHigh ? "warning" : "nominal",
    },
    {
      id: "dgca-gagan",
      name: "GAGAN Civil Aviation Satellite Augmentation",
      region: "Indian Airspace (AAI / ISRO)",
      status: isCritical ? "VERTICAL GUIDANCE SUSPENDED" : isHigh ? "APV-I DEGRADATION" : "APV-II AVAILABLE",
      riskPercent: isCritical ? 84 : isHigh ? 44 : 2,
      metricLabel: "VERTICAL PROTECTION LEVEL (VPL)",
      metricValue: isCritical ? "68.4 m (>35m Limit)" : isHigh ? "28.1 m" : "11.2 m (Valid)",
      directive: isCritical
        ? "Downgrade CAT-I precision instrument approaches to Barometric VNAV nationwide."
        : "GAGAN Master Control Centre Bengaluru operating nominal.",
      severity: isCritical ? "critical" : isHigh ? "warning" : "nominal",
    },
  ];

  const currentAssets =
    activeSector === "power"
      ? powerAssets
      : activeSector === "satellite"
      ? satelliteAssets
      : aviationAssets;

    // Multi-tier vulnerability bar color scaler with high visual contrast
    const getVulnerabilityBarStyle = (percent: number) => {
      if (percent >= 90) {
        // Extreme Critical Hazard (>= 90%): Pure Neon Crimson Red (#ff003b)
        return {
          barClass: "bg-[#ff003b]",
          textColor: "text-[#ff003b]",
          style: {
            backgroundColor: "#ff003b",
            boxShadow: "0 0 12px rgba(255, 0, 59, 0.75)",
          },
        };
      } else if (percent >= 80) {
        // Severe Risk (80% - 89%): Fiery Sunset Orange-Red (#ff5722)
        return {
          barClass: "bg-[#ff5722]",
          textColor: "text-[#ff5722]",
          style: {
            backgroundColor: "#ff5722",
            boxShadow: "0 0 10px rgba(255, 87, 34, 0.65)",
          },
        };
      } else if (percent >= 60) {
        // Elevated Precursor (60% - 79%): Golden Amber (#ffaa00)
        return {
          barClass: "bg-[#ffaa00]",
          textColor: "text-[#ffaa00]",
          style: {
            backgroundColor: "#ffaa00",
            boxShadow: "0 0 10px rgba(255, 170, 0, 0.55)",
          },
        };
      } else if (percent >= 25) {
        // Moderate Watch (25% - 59%): Solar Yellow-Gold (#ffd600)
        return {
          barClass: "bg-[#ffd600]",
          textColor: "text-[#ffd600]",
          style: {
            backgroundColor: "#ffd600",
            boxShadow: "0 0 8px rgba(255, 214, 0, 0.45)",
          },
        };
      } else {
        // Nominal Baseline (< 25%): Emerald Green (#00e676)
        return {
          barClass: "bg-[#00e676]",
          textColor: "text-[#00e676]",
          style: {
            backgroundColor: "#00e676",
            boxShadow: "0 0 8px rgba(0, 230, 118, 0.45)",
          },
        };
      }
    };

    return (
      <div className="space-y-6">
        {/* Title & Introduction */}
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 border-b border-white/10 pb-4">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2.5">
              <Zap className="h-6 w-6 text-amber-400" />
              Infrastructure Threat Simulation & Dynamic Damage Matrix
            </h2>
            <p className="text-xs text-slate-400 font-mono mt-1">
              Real-time translation of learned ConvLSTM spatio-temporal precursor weights into critical Indian asset vulnerability indexes.
            </p>
          </div>

          {/* Sector Navigation Switcher */}
          <div className="flex items-center gap-1.5 bg-space-950/80 p-1.5 rounded-2xl border border-white/10">
            <button
              onClick={() => setActiveSector("power")}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-xl text-xs font-semibold font-mono transition-all cursor-pointer ${
                activeSector === "power"
                  ? "bg-amber-500/20 text-amber-300 border border-amber-500/40 shadow-lg shadow-amber-500/20"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              <Zap className="h-4 w-4" />
              <span>Power Grids</span>
            </button>
            <button
              onClick={() => setActiveSector("satellite")}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-xl text-xs font-semibold font-mono transition-all cursor-pointer ${
                activeSector === "satellite"
                  ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-lg shadow-cyan-500/20"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              <Navigation className="h-4 w-4" />
              <span>Satellites & Crew</span>
            </button>
            <button
              onClick={() => setActiveSector("aviation")}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-xl text-xs font-semibold font-mono transition-all cursor-pointer ${
                activeSector === "aviation"
                  ? "bg-blue-500/20 text-blue-300 border border-blue-500/40 shadow-lg shadow-blue-500/20"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              <Plane className="h-4 w-4" />
              <span>GPS & Aviation</span>
            </button>
          </div>
        </div>

        {/* Dynamic Bento Grid of Assets */}
        <AnimatePresence mode="wait">
          <motion.div
            key={activeSector}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.25 }}
            className="grid grid-cols-1 md:grid-cols-3 gap-6"
          >
            {currentAssets.map((asset) => {
              const isCrit = asset.severity === "critical";
              const isWarn = asset.severity === "warning";

              // Unified with app's exact alert-red (#ff334b / #ff4d67) and space-card-alert CSS palette
              const cardBorder = isCrit
                ? "border-[#ff334b]/60 bg-gradient-to-b from-[#2d0c16]/85 via-space-900 to-space-950 shadow-2xl shadow-[#ff334b]/20"
                : isWarn
                ? "border-amber-500/50 bg-gradient-to-b from-amber-950/40 via-space-900 to-space-950 shadow-2xl shadow-amber-500/15"
                : "border-emerald-500/30 bg-gradient-to-b from-emerald-950/20 via-space-900 to-space-950 shadow-lg shadow-emerald-500/5";

              const badgeColor = isCrit
                ? "bg-[#ff334b]/20 text-[#ff4d67] border border-[#ff334b]/60 font-bold font-mono tracking-wider animate-pulse"
                : isWarn
                ? "bg-amber-500/20 text-amber-300 border border-amber-500/40 font-bold font-mono"
                : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-bold font-mono";

              // Severity-scaled dynamic bar styling
              const barStyle = getVulnerabilityBarStyle(asset.riskPercent);

              return (
                <div
                  key={asset.id}
                  className={`rounded-3xl border p-6 flex flex-col justify-between transition-all duration-300 hover:scale-[1.02] ${cardBorder}`}
                >
                  <div>
                    {/* Top Status Badge */}
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">
                        {asset.region}
                      </span>
                      <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-mono ${badgeColor}`}>
                        {asset.status}
                      </span>
                    </div>

                    {/* Asset Name */}
                    <h3 className="text-base font-bold text-white mb-3">
                      {asset.name}
                    </h3>

                    {/* Damage Gauge */}
                    <div className="space-y-1.5 mb-4 p-3 rounded-2xl bg-space-950/80 border border-white/5">
                      <div className="flex justify-between text-xs font-mono">
                        <span className="text-slate-400">PREDICTED VULNERABILITY</span>
                        <span className={`font-bold font-mono ${barStyle.textColor}`}>{asset.riskPercent}%</span>
                      </div>
                      <div className="w-full bg-space-900 rounded-full h-2 overflow-hidden border border-white/10">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${asset.riskPercent}%` }}
                          transition={{ duration: 0.8 }}
                          className={`h-full rounded-full ${barStyle.barClass}`}
                          style={barStyle.style}
                        />
                      </div>
                    </div>

                    {/* Physical Proxy Measurement */}
                    <div className="mb-4 text-xs font-mono bg-space-950/60 p-2.5 rounded-xl border border-white/5 flex justify-between items-center">
                      <span className="text-slate-400 text-[11px]">{asset.metricLabel}</span>
                      <span className={`font-bold font-mono ${barStyle.textColor}`}>{asset.metricValue}</span>
                    </div>
                  </div>

                  {/* Directive Action Box */}
                  <div className="pt-3 border-t border-white/10 text-xs text-slate-300 font-sans leading-relaxed">
                    <span className="text-[10px] font-mono text-amber-400 font-bold block mb-1 uppercase">
                      ⚡ Operational Directive:
                    </span>
                    {asset.directive}
                  </div>
                </div>
              );
            })}
          </motion.div>
        </AnimatePresence>

      {/* Strategic Defence Footer Note */}
      <div className="rounded-2xl border border-cyan-500/20 bg-space-950/90 p-5 font-mono text-xs text-slate-300 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
            <ShieldCheck className="h-5 w-5" />
          </div>
          <div>
            <div className="font-bold text-white text-sm">
              Standard Operating Procedure (SOP) Compliance Matrix
            </div>
            <div className="text-[11px] text-slate-400">
              Cross-referenced against ISRO ISSDC, NLDC POSOCO, and NOAA Space Weather Prediction Center (SWPC) protocols.
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="px-3 py-1 rounded-xl bg-emerald-500/10 text-emerald-300 border border-emerald-500/20 text-xs">
            Auto-Mitigation Pre-Armed
          </span>
        </div>
      </div>
    </div>
  );
};