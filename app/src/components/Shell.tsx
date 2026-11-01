"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { type Club, useApi } from "@/lib/api";
import { VIEWS } from "@/lib/views";
import { CommandPalette } from "./CommandPalette";

/** Re-skins the app from the club config: switching club switches colours. */
function useClubTheme(club: Club | undefined) {
  useEffect(() => {
    const theme = club?.theme;
    if (!theme) return;
    const root = document.documentElement.style;
    root.setProperty("--club", theme.accent);
    if (theme.secondary) root.setProperty("--club-2", theme.secondary);
  }, [club]);
}

export function Shell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { data: club, error } = useApi<Club>("/api/club");
  const [paletteOpen, setPaletteOpen] = useState(false);
  useClubTheme(club);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen((open) => !open);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="flex min-h-screen flex-col md:flex-row">
      <nav
        aria-label="Views"
        className="order-last flex border-t border-line bg-night md:order-first md:w-44 md:flex-col md:border-t-0 md:border-r md:py-6"
      >
        <div className="hidden px-5 pb-8 md:block">
          <div className="font-display text-2xl leading-none font-black tracking-wide uppercase">
            {club?.club ?? "Recruitment"}
          </div>
          <div className="mt-1 font-mono text-[10px] text-fog uppercase">
            {club ? `${club.season_label} · ${club.formation ?? "no formation"}` : "loading"}
          </div>
        </div>
        {VIEWS.map((v) => {
          const active = v.href === "/" ? pathname === "/" : pathname.startsWith(v.href);
          return (
            <Link
              key={v.href}
              href={v.href}
              aria-current={active ? "page" : undefined}
              className={`flex-1 border-t-2 px-2 py-3 text-center text-sm transition-colors md:flex-none md:border-t-0 md:border-l-2 md:px-5 md:py-2.5 md:text-left ${
                active
                  ? "border-club text-chalk"
                  : "border-transparent text-fog hover:text-chalk"
              }`}
            >
              {v.label}
            </Link>
          );
        })}
        <button
          type="button"
          onClick={() => setPaletteOpen(true)}
          className="mt-auto hidden px-5 py-3 text-left font-mono text-[11px] text-fog hover:text-chalk md:block"
        >
          Find a player <kbd className="ml-1 rounded border border-line px-1">Ctrl K</kbd>
        </button>
      </nav>

      <main className="min-w-0 flex-1">
        {error && (
          <div role="alert" className="border-b border-club/60 bg-club/10 px-6 py-3 text-sm">
            {error.message} Start it from the repo root with{" "}
            <code className="font-mono text-xs">
              .venv/Scripts/uvicorn recruit_intel.api.main:app --port 8000
            </code>
          </div>
        )}
        {children}
      </main>

      <CommandPalette open={paletteOpen} onOpenChange={setPaletteOpen} />
    </div>
  );
}
