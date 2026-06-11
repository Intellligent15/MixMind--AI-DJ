import { NavHeader } from "@/components/NavHeader";
import { Player } from "@/components/Player";
import { TransitionControls } from "@/components/TransitionControls";

export default function PlayerPage() {
  return (
    <main className="min-h-screen max-w-3xl mx-auto p-8 flex flex-col gap-8 font-mono">
      <NavHeader subtitle="Playing" />
      <Player />
      <TransitionControls />
    </main>
  );
}
