import { NavHeader } from "@/components/NavHeader";
import { ProcessingView } from "@/components/ProcessingView";

export default function ProcessingPage() {
  return (
    <main className="min-h-screen max-w-3xl mx-auto p-8 flex flex-col gap-8 font-mono">
      <NavHeader subtitle="Processing" />
      <ProcessingView />
    </main>
  );
}
