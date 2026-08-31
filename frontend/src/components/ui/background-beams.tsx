import { useEffect, useRef } from "react";
import { cn } from "../../lib/utils";

export const BackgroundBeams = ({ className }: { className?: string }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationFrameId: number;
    let width = (canvas.width = window.innerWidth);
    let height = (canvas.height = window.innerHeight);

    const handleResize = () => {
      if (!canvas) return;
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    };
    window.addEventListener("resize", handleResize);

    // Generate stars and nebula particles
    const particles: Array<{
      x: number;
      y: number;
      radius: number;
      color: string;
      vx: number;
      vy: number;
      alpha: number;
      alphaSpeed: number;
    }> = [];

    const colors = ["#00e5ff", "#42a5f5", "#ffb300", "#ffffff", "#7c4dff"];
    for (let i = 0; i < 90; i++) {
      particles.push({
        x: Math.random() * width,
        y: Math.random() * height,
        radius: Math.random() * 1.8 + 0.5,
        color: colors[Math.floor(Math.random() * colors.length)],
        vx: (Math.random() - 0.5) * 0.25,
        vy: (Math.random() - 0.5) * 0.25,
        alpha: Math.random() * 0.8 + 0.2,
        alphaSpeed: (Math.random() - 0.5) * 0.008,
      });
    }

    const render = () => {
      ctx.clearRect(0, 0, width, height);

      // Deep space radial background glow
      const grad = ctx.createRadialGradient(
        width * 0.5,
        height * 0.15,
        50,
        width * 0.5,
        height * 0.5,
        width * 0.8
      );
      grad.addColorStop(0, "rgba(10, 20, 50, 0.4)");
      grad.addColorStop(0.5, "rgba(5, 9, 25, 0.7)");
      grad.addColorStop(1, "rgba(2, 4, 12, 0.95)");
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, width, height);

      // Render connected cosmic grid / beams
      particles.forEach((p, idx) => {
        p.x += p.vx;
        p.y += p.vy;
        p.alpha += p.alphaSpeed;
        if (p.alpha <= 0.1 || p.alpha >= 0.9) p.alphaSpeed = -p.alphaSpeed;

        if (p.x < 0) p.x = width;
        if (p.x > width) p.x = 0;
        if (p.y < 0) p.y = height;
        if (p.y > height) p.y = 0;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.globalAlpha = p.alpha;
        ctx.shadowBlur = 8;
        ctx.shadowColor = p.color;
        ctx.fill();

        // Connect near particles with subtle glowing beams
        for (let j = idx + 1; j < particles.length; j++) {
          const p2 = particles[j];
          const dist = Math.hypot(p.x - p2.x, p.y - p2.y);
          if (dist < 90) {
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.strokeStyle = p.color;
            ctx.globalAlpha = (1 - dist / 90) * 0.12;
            ctx.lineWidth = 0.6;
            ctx.stroke();
          }
        }
      });
      ctx.globalAlpha = 1.0;

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      window.removeEventListener("resize", handleResize);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className={cn("fixed inset-0 pointer-events-none z-0", className)}
    />
  );
};
