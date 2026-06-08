"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { api, isStatusError, type Queue } from "@/lib/api";

type Props = {
  subtitle?: string;
};

export function NavHeader({ subtitle }: Props) {
  // Reuses the same ["queue","current"] cache that QueueBuilder / Player
  // populate — no extra network when those pages are mounted.
  const queue = useQuery<Queue | null>({
    queryKey: ["queue", "current"],
    queryFn: async () => {
      try {
        return await api.getCurrentQueue();
      } catch (err) {
        if (isStatusError(err, 404)) return null;
        throw err;
      }
    },
  });

  const hasLocked = queue.data?.locked === true;

  return (
    <header className="flex items-baseline justify-between">
      <h1 className="text-3xl font-bold">
        <Link href="/">MixMind</Link>
      </h1>
      <nav className="flex items-baseline gap-4 text-sm">
        <Link href="/" className="hover:underline">
          Queue
        </Link>
        {hasLocked && (
          <>
            <Link href="/processing" className="hover:underline">
              Processing
            </Link>
            <Link href="/player" className="hover:underline">
              Player
            </Link>
          </>
        )}
        <Link href="/library" className="hover:underline">
          Library
        </Link>
        {subtitle && <span className="opacity-70 text-xs">{subtitle}</span>}
      </nav>
    </header>
  );
}
