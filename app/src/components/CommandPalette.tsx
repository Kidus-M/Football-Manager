"use client";

import { Command } from "cmdk";
import { useRouter } from "next/navigation";
import { useDeferredValue, useState } from "react";
import { league, type PlayerHit, query, useApi } from "@/lib/api";
import { VIEWS } from "@/lib/views";

/** Ctrl/Cmd+K: jump to any player or view. */
export function CommandPalette({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const router = useRouter();
  const [search, setSearch] = useState("");
  const deferred = useDeferredValue(search.trim());
  const { data: hits = [], isLoading } = useApi<PlayerHit[]>(
    deferred.length >= 2 ? `/api/players${query({ q: deferred, limit: 8 })}` : null,
  );

  const go = (href: string) => {
    onOpenChange(false);
    setSearch("");
    router.push(href);
  };

  return (
    <Command.Dialog
      open={open}
      onOpenChange={onOpenChange}
      label="Find a player or view"
      shouldFilter={false}
      overlayClassName="fixed inset-0 bg-night/70 backdrop-blur-[2px]"
      contentClassName="fixed top-[14vh] left-1/2 w-[min(560px,calc(100vw-32px))] -translate-x-1/2 overflow-hidden rounded-lg border border-line bg-turf shadow-2xl"
    >
      <Command.Input
        value={search}
        onValueChange={setSearch}
        placeholder="Player name, or a view"
        className="w-full border-b border-line bg-transparent px-4 py-3.5 text-base outline-none placeholder:text-fog"
      />
      <Command.List className="max-h-[50vh] overflow-y-auto p-1.5">
        <Command.Empty className="px-3 py-6 text-center text-sm text-fog">
          {isLoading ? "Searching…" : deferred.length < 2 ? "Type two letters to search players." : "No player by that name in the top five leagues."}
        </Command.Empty>
        {hits.length > 0 && (
          <Command.Group heading="Players" className="[&_[cmdk-group-heading]]:px-3 [&_[cmdk-group-heading]]:py-1.5 [&_[cmdk-group-heading]]:font-mono [&_[cmdk-group-heading]]:text-[10px] [&_[cmdk-group-heading]]:text-fog [&_[cmdk-group-heading]]:uppercase">
            {hits.map((p) => (
              <Command.Item
                key={p.player_id}
                value={p.player_id}
                onSelect={() => go(`/compare?ids=${p.player_id}`)}
                className="flex cursor-pointer items-baseline justify-between gap-3 rounded px-3 py-2 data-[selected=true]:bg-turf-2"
              >
                <span>{p.name}</span>
                <span className="font-mono text-[11px] text-fog">
                  {[p.group, p.club, league(p.competition_id)].filter(Boolean).join(" · ")}
                </span>
              </Command.Item>
            ))}
          </Command.Group>
        )}
        <Command.Group heading="Views" className="[&_[cmdk-group-heading]]:px-3 [&_[cmdk-group-heading]]:py-1.5 [&_[cmdk-group-heading]]:font-mono [&_[cmdk-group-heading]]:text-[10px] [&_[cmdk-group-heading]]:text-fog [&_[cmdk-group-heading]]:uppercase">
          {VIEWS.filter((v) => !deferred || v.label.toLowerCase().includes(deferred.toLowerCase())).map((v) => (
            <Command.Item
              key={v.href}
              value={v.href}
              onSelect={() => go(v.href)}
              className="flex cursor-pointer items-baseline justify-between rounded px-3 py-2 data-[selected=true]:bg-turf-2"
            >
              <span>{v.label}</span>
              <span className="text-xs text-fog">{v.hint}</span>
            </Command.Item>
          ))}
        </Command.Group>
      </Command.List>
    </Command.Dialog>
  );
}
