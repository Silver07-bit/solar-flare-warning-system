"""
🛡️ Space Weather Decision Support & Extensible CME Transit Module
Translates physical deep learning flare forecasts into actionable defense protocols
using official NOAA Space Weather Scales (R1-R5, S1-S5, G1-G5):
  - R-Scale: Radio Blackouts (Solar X-ray emission)
  - S-Scale: Solar Radiation Storms (Energetic Solar Protons)
  - G-Scale: Geomagnetic Storms (Coronal Mass Ejections & Interplanetary Shock)
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone, timedelta


class SpaceWeatherDecisionEngine:
    """
    Decoupled Decision-Support Engine.
    Maps ML flare forecasts to national infrastructure threat levels.
    """

    @staticmethod
    def map_flare_to_noaa_scales(flare_class: str, peak_flux_wm2: float) -> Dict[str, Any]:
        """
        Maps peak flux to standard NOAA R-Scale (Radio Blackout).
        """
        if flare_class.startswith("X") or peak_flux_wm2 >= 1.0e-4:
            r_scale = "R3 - Strong to R5 - Extreme"
            hf_impact = "Complete HF radio blackout on Sunlit side of Earth (1-2 hours). Loss of radio contact."
            gnss_impact = "Significant L-band ionospheric scintillation; NavIC positional errors degraded."
        elif flare_class.startswith("M") or peak_flux_wm2 >= 1.0e-5:
            r_scale = "R1 - Minor to R2 - Moderate"
            hf_impact = "Limited blackout of HF radio communication on Sunlit side, loss of radio contact for tens of minutes."
            gnss_impact = "Minor degraded satellite navigation signals."
        elif flare_class.startswith("C"):
            r_scale = "R0 - Baseline"
            hf_impact = "Nominal HF propagation."
            gnss_impact = "Nominal GNSS positioning."
        else:
            r_scale = "R0 - Nominal"
            hf_impact = "Zero space-weather disturbance."
            gnss_impact = "Nominal GNSS positioning (< 2.5m error)."

        return {
            "r_scale": r_scale,
            "hf_radio_impact": hf_impact,
            "gnss_navigation_impact": gnss_impact
        }

    @staticmethod
    def estimate_cme_transit(
        cme_associated: bool = True,
        cme_speed_km_s: Optional[float] = None,
        flare_class: str = "X1.0"
    ) -> Dict[str, Any]:
        """
        Calculates estimated CME Earth-transit duration.
        If no validated CME event is associated, explicitly reports unavailable.
        """
        if not cme_associated:
            return {
                "cme_status": "NONE_DETECTED",
                "message": "CME impact estimation unavailable for this event. No correlated CME ejecta detected in coronagraph.",
                "transit_time_hours": None,
                "estimated_arrival": None
            }

        # Estimate speed if not provided
        if cme_speed_km_s is None:
            if flare_class.startswith("X"):
                cme_speed_km_s = 1400.0  # Fast CME
            elif flare_class.startswith("M"):
                cme_speed_km_s = 750.0   # Moderate CME
            else:
                cme_speed_km_s = 450.0   # Slow CME

        # Sun-Earth distance ~ 149.6 million km
        # Empirical interplanetary deceleration model: t = d / (v * 0.85)
        transit_seconds = (149.6e6) / (cme_speed_km_s * 0.85)
        transit_hours = transit_seconds / 3600.0

        arrival_time = datetime.now(timezone.utc) + timedelta(hours=transit_hours)

        return {
            "cme_status": "CORRELATED_CME_DETECTED",
            "cme_speed_km_s": round(cme_speed_km_s, 1),
            "estimated_transit_hours": f"{transit_hours:.1f} Hours ({int(transit_hours-6)}-{int(transit_hours+6)}h window)",
            "estimated_earth_impact_utc": arrival_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "geomagnetic_g_scale": "G3 - Strong to G4 - Severe" if cme_speed_km_s > 1000 else "G1 - Minor to G2 - Moderate"
        }

    @staticmethod
    def generate_national_infrastructure_directives(
        flare_prob: float,
        flare_class: str,
        peak_flux: float
    ) -> list:
        """
        Generates actionable directives for ISRO NavIC, GSAT, PGCIL, and DGCA Civil Aviation.
        """
        is_high_risk = flare_prob >= 55.0 or flare_class.startswith("X")
        is_moderate_risk = flare_prob >= 35.0 or flare_class.startswith("M")

        directives = []

        # 1. ISRO NavIC (IRNSS) Constellation
        if is_high_risk:
            directives.append({
                "sector": "ISRO NavIC (IRNSS Constellation)",
                "status": "CRITICAL RISK",
                "level": "RED",
                "directive": "Broadcast differential ionospheric delay compensation flags to ground receivers. Potential pseudo-range error elevation."
            })
        elif is_moderate_risk:
            directives.append({
                "sector": "ISRO NavIC (IRNSS Constellation)",
                "status": "ELEVATED WATCH",
                "level": "AMBER",
                "directive": "Monitor L5 and S-band carrier-to-noise ratio. Maintain tracking over Indian subcontinent."
            })
        else:
            directives.append({
                "sector": "ISRO NavIC (IRNSS Constellation)",
                "status": "NOMINAL",
                "level": "GREEN",
                "directive": "Standard positional accuracy (< 2.5m) maintained across all 7 constellation satellites."
            })

        # 2. GSAT / INSAT Geostationary Telecom
        if is_high_risk:
            directives.append({
                "sector": "GSAT / INSAT Geostationary Telecom",
                "status": "SURGE SAFE-MODE",
                "level": "RED",
                "directive": "Place sensitive GEO transponders into surge protection. Orient solar panels away from deep dielectric charging vectors."
            })
        else:
            directives.append({
                "sector": "GSAT / INSAT Geostationary Telecom",
                "status": "NOMINAL",
                "level": "GREEN",
                "directive": "Nominal satellite power bus and downlink SNR across Ku/Ka-band transponders."
            })

        # 3. Indian Power Grid (PGCIL / POSOCO)
        if is_high_risk:
            directives.append({
                "sector": "Indian Power Grid (PGCIL / POSOCO)",
                "status": "GIC WARNING",
                "level": "RED",
                "directive": "Engage series capacitor banks across Northern and Western 765kV/400kV transmission corridors to block Geomagnetically Induced Currents."
            })
        elif is_moderate_risk:
            directives.append({
                "sector": "Indian Power Grid (PGCIL / POSOCO)",
                "status": "GIC WATCH",
                "level": "AMBER",
                "directive": "Alert Regional Load Despatch Centres (RLDCs). Monitor transformer neutral DC ground currents."
            })
        else:
            directives.append({
                "sector": "Indian Power Grid (PGCIL / POSOCO)",
                "status": "NOMINAL",
                "level": "GREEN",
                "directive": "Nominal ground potential baseline. Zero transformer saturation hazard."
            })

        # 4. Gaganyaan Manned Spaceflight Mission
        if is_high_risk:
            directives.append({
                "sector": "Gaganyaan Human Spaceflight Program",
                "status": "EVA NO-GO",
                "level": "RED",
                "directive": "Astronaut radiation dose rate elevated in Low Earth Orbit (LEO). Extravehicular Activity (EVA) strictly prohibited."
            })
        else:
            directives.append({
                "sector": "Gaganyaan Human Spaceflight Program",
                "status": "SAFE",
                "level": "GREEN",
                "directive": "Safe orbital radiation environment. Background galactic cosmic ray levels within baseline thresholds."
            })

        return directives
