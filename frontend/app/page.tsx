import { NavHeader } from "@/components/NavHeader";
import { QueueBuilder } from "@/components/QueueBuilder";

export default function HomePage() {
  return (
    <main className="min-h-screen max-w-5xl mx-auto p-8 flex flex-col gap-8 font-mono">
      <NavHeader />
      <QueueBuilder />
    </main>
  );
}
