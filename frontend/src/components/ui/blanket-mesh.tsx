import { useEffect, useRef } from "react";
import { cn } from "../../lib/utils";

interface BlanketMeshProps {
  className?: string;
  gridGap?: number;
  isFlareActive?: boolean;
}

interface Point {
  x: number;
  y: number;
  originX: number;
  originY: number;
  vx: number;
  vy: number;
  z: number;
  vz: number;
}

interface Stardust {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  alpha: number;
  life: number;
  maxLife: number;
  color: string;
}

interface Shockwave {
  x: number;
  y: number;
  radius: number;
  maxRadius: number;
  intensity: number;
}

export const BlanketMesh = ({
  className,
  gridGap = 32,
  isFlareActive = false,
}: BlanketMeshProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animId: number;
    let width = (canvas.width = window.innerWidth);
    let height = (canvas.height = window.innerHeight);

    let points: Point[][] = [];
    let cols = Math.ceil(width / gridGap) + 1;
    let rows = Math.ceil(height / gridGap) + 1;

    let particles: Stardust[] = [];
    let shockwaves: Shockwave[] = [];

    // Background cosmic stars
    const stars: Array<{ x: number; y: number; r: number; alpha: number; speed: number }> = [];
    for (let i = 0; i < 70; i++) {
      stars.push({
        x: Math.random() * width,
        y: Math.random() * height,
        r: Math.random() * 1.5 + 0.5,
        alpha: Math.random() * 0.7 + 0.3,
        speed: (Math.random() - 0.5) * 0.005,
      });
    }

    const initGrid = () => {
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
      cols = Math.ceil(width / gridGap) + 1;
      rows = Math.ceil(height / gridGap) + 1;
      points = [];

      for (let r = 0; r < rows; r++) {
        const row: Point[] = [];
        for (let c = 0; c < cols; c++) {
          const x = c * gridGap;
          const y = r * gridGap;
          row.push({
            x,
            y,
            originX: x,
            originY: y,
            vx: 0,
            vy: 0,
            z: 0,
            vz: 0,
          });
        }
        points.push(row);
      }
    };

    initGrid();

    // Mouse tracking for gravitational sinkhole
    const mouse = {
      x: -1000,
      y: -1000,
      prevX: -1000,
      prevY: -1000,
      speedX: 0,
      speedY: 0,
    };

    const onMouseMove = (e: MouseEvent) => {
      const currentX = e.clientX;
      const currentY = e.clientY;

      mouse.speedX = currentX - (mouse.prevX === -1000 ? currentX : mouse.prevX);
      mouse.speedY = currentY - (mouse.prevY === -1000 ? currentY : mouse.prevY);
      mouse.prevX = mouse.x = currentX;
      mouse.prevY = mouse.y = currentY;

      // Spawn stardust sparks on mouse motion
      const speed = Math.hypot(mouse.speedX, mouse.speedY);
      if (speed > 1.5 && particles.length < 80) {
        const pColors = ["#00e5ff", "#00b0ff", "#38bdf8", "#00e676", "#ffffff"];
        for (let i = 0; i < Math.min(4, Math.floor(speed / 3) + 1); i++) {
          particles.push({
            x: currentX + (Math.random() - 0.5) * 16,
            y: currentY + (Math.random() - 0.5) * 16,
            vx: mouse.speedX * 0.15 + (Math.random() - 0.5) * 1.5,
            vy: mouse.speedY * 0.15 + (Math.random() - 0.5) * 1.5,
            radius: Math.random() * 2 + 0.8,
            alpha: 1,
            life: 0,
            maxLife: Math.random() * 28 + 20,
            color: pColors[Math.floor(Math.random() * pColors.length)],
          });
        }
      }
    };

    const onMouseDown = (e: MouseEvent) => {
      shockwaves.push({
        x: e.clientX,
        y: e.clientY,
        radius: 12,
        maxRadius: 360,
        intensity: 36,
      });
    };

    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mousedown", onMouseDown);
    window.addEventListener("resize", initGrid);

    const sinkRadius = 180;
    const maxDepth = 40;
    const recoverySpeed = 0.22; // Split-second smooth return (no wavy ringing)

    const render = () => {
      ctx.clearRect(0, 0, width, height);

      // 1. Draw Space Background Nebula in Deep Constant Dark Navy
      const nebulaGrad = ctx.createRadialGradient(
        width * 0.5,
        height * 0.3,
        50,
        width * 0.5,
        height * 0.5,
        width * 0.75
      );
      nebulaGrad.addColorStop(0, "rgba(4, 12, 34, 0.98)");
      nebulaGrad.addColorStop(0.5, "rgba(3, 8, 22, 0.99)");
      nebulaGrad.addColorStop(1, "rgba(2, 6, 23, 1)");
      ctx.fillStyle = nebulaGrad;
      ctx.fillRect(0, 0, width, height);

      // 2. Draw Twinkling Background Stars
      stars.forEach((s) => {
        s.alpha += s.speed;
        if (s.alpha > 0.9 || s.alpha < 0.2) s.speed = -s.speed;
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 255, 255, ${Math.max(0.1, s.alpha)})`;
        ctx.fill();
      });

      // 3. Process Shockwaves
      for (let sIdx = shockwaves.length - 1; sIdx >= 0; sIdx--) {
        const sw = shockwaves[sIdx];
        sw.radius += 10;
        sw.intensity *= 0.92;

        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < cols; c++) {
            const p = points[r][c];
            const dx = p.originX - sw.x;
            const dy = p.originY - sw.y;
            const dist = Math.hypot(dx, dy);
            const ringDist = Math.abs(dist - sw.radius);

            if (ringDist < 40 && dist > 0) {
              const force = (1 - ringDist / 40) * sw.intensity;
              p.x += (dx / dist) * force * 0.25;
              p.y += (dy / dist) * force * 0.25;
              p.z += force * 0.35;
            }
          }
        }

        if (sw.radius > sw.maxRadius || sw.intensity < 0.5) {
          shockwaves.splice(sIdx, 1);
        }
      }

      // 4. Update Crisp 3D Sink-In Physics with Split-Second Instant Flat Recovery (Zero Wavy Nature)
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const p = points[r][c];

          const dx = p.originX - mouse.x;
          const dy = p.originY - mouse.y;
          const dist = Math.hypot(dx, dy);

          let targetX = p.originX;
          let targetY = p.originY;
          let targetZ = 0;

          if (dist < sinkRadius && dist > 0) {
            const factor = Math.pow(1 - dist / sinkRadius, 2.0);
            targetZ = factor * maxDepth;
            const inwardPull = factor * 10;
            targetX = p.originX - (dx / dist) * inwardPull;
            targetY = p.originY - (dy / dist) * inwardPull;
          }

          // Direct smooth critically damped snap back (returns flat in a split-second with zero waves)
          p.x += (targetX - p.x) * recoverySpeed;
          p.y += (targetY - p.y) * recoverySpeed;
          p.z += (targetZ - p.z) * recoverySpeed;
        }
      }

      // 5. Draw 3D Gravitational Well Depressed Shadow & Subtle Rim
      if (mouse.x > -500) {
        const sinkGrad = ctx.createRadialGradient(
          mouse.x,
          mouse.y,
          0,
          mouse.x,
          mouse.y,
          sinkRadius * 0.95
        );
        sinkGrad.addColorStop(0, "rgba(1, 6, 18, 0.55)");
        sinkGrad.addColorStop(0.5, "rgba(0, 229, 255, 0.06)");
        sinkGrad.addColorStop(1, "rgba(3, 7, 18, 0)");

        ctx.fillStyle = sinkGrad;
        ctx.beginPath();
        ctx.arc(mouse.x, mouse.y, sinkRadius * 0.95, 0, Math.PI * 2);
        ctx.fill();

        // 2 Delicate depth contour rings in the sinkhole
        for (let i = 1; i <= 2; i++) {
          const rRing = (sinkRadius / 2.8) * i;
          ctx.beginPath();
          ctx.arc(mouse.x, mouse.y, rRing, 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(0, 229, 255, ${0.18 - i * 0.05})`;
          ctx.lineWidth = 0.8;
          ctx.stroke();
        }
      }

      // 6. Draw Mesh Lines with 3D Depth Curve (Subtle and non-distracting)
      const defaultLineColor = "rgba(0, 229, 255, 0.08)";

      ctx.lineWidth = 0.8;

      // Horizontal lines
      for (let r = 0; r < rows; r++) {
        ctx.beginPath();
        ctx.moveTo(points[r][0].x, points[r][0].y);
        for (let c = 1; c < cols; c++) {
          const p = points[r][c];
          ctx.lineTo(p.x, p.y);
        }
        ctx.strokeStyle = defaultLineColor;
        ctx.stroke();
      }

      // Vertical lines
      for (let c = 0; c < cols; c++) {
        ctx.beginPath();
        ctx.moveTo(points[0][c].x, points[0][c].y);
        for (let r = 1; r < rows; r++) {
          const p = points[r][c];
          ctx.lineTo(p.x, p.y);
        }
        ctx.strokeStyle = defaultLineColor;
        ctx.stroke();
      }

      // 7. Draw Deeply Sunk & Gentle Subtle Tension Nodes (Less bright & non-intrusive)
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const p = points[r][c];
          const disp = Math.hypot(p.x - p.originX, p.y - p.originY, p.z);

          if (disp > 1.8) {
            const glowAlpha = Math.min(0.35, disp / 25);
            ctx.beginPath();
            ctx.arc(p.x, p.y, Math.min(2.0, 0.7 + disp * 0.06), 0, Math.PI * 2);
            ctx.fillStyle = isFlareActive
              ? `rgba(244, 63, 94, ${glowAlpha})`
              : `rgba(56, 189, 248, ${glowAlpha})`;
            ctx.shadowBlur = Math.min(3, disp * 0.3);
            ctx.shadowColor = isFlareActive ? "#f43f5e" : "#38bdf8";
            ctx.fill();
            ctx.shadowBlur = 0;
          }
        }
      }

      // 8. Draw Cursor Particle Trail
      for (let pIdx = particles.length - 1; pIdx >= 0; pIdx--) {
        const pt = particles[pIdx];
        pt.life++;
        pt.x += pt.vx;
        pt.y += pt.vy;
        pt.vx *= 0.95;
        pt.vy *= 0.95;
        pt.alpha = 1 - pt.life / pt.maxLife;

        ctx.beginPath();
        ctx.arc(pt.x, pt.y, pt.radius, 0, Math.PI * 2);
        ctx.fillStyle = pt.color;
        ctx.globalAlpha = pt.alpha * 0.9;
        ctx.shadowBlur = 10;
        ctx.shadowColor = pt.color;
        ctx.fill();
        ctx.globalAlpha = 1.0;
        ctx.shadowBlur = 0;

        if (pt.life >= pt.maxLife) {
          particles.splice(pIdx, 1);
        }
      }

      // Bleed mouse velocity
      mouse.speedX *= 0.82;
      mouse.speedY *= 0.82;

      animId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mousedown", onMouseDown);
      window.removeEventListener("resize", initGrid);
    };
  }, [gridGap, isFlareActive]);

  return (
    <canvas
      ref={canvasRef}
      className={cn(
        "fixed inset-0 pointer-events-none z-0 h-full w-full opacity-90 transition-opacity duration-700",
        className
      )}
    />
  );
};