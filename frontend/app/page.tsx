import { SearchPanel } from "@/components/SearchPanel";
import { DownloadedSongs } from "@/components/DownloadedSongs";

export default function HomePage() {
  return (
    <main className="min-h-screen max-w-4xl mx-auto p-8 flex flex-col gap-8 font-mono">
      <header className="flex items-baseline justify-between">
        <h1 className="text-3xl font-bold">AI DJ</h1>
        <p className="text-xs opacity-70">Phase 3 — analysis pipeline</p>
      </header>
      <SearchPanel />
      <DownloadedSongs />
    </main>
  );
}
