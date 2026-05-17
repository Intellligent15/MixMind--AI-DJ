import { NavHeader } from "@/components/NavHeader";

export default function HomePage() {
  return (
    <main className="min-h-screen max-w-5xl mx-auto p-8 flex flex-col gap-8 font-mono">
      <NavHeader subtitle="Phase 4 — queue" />
      <p className="text-sm opacity-70">
        Queue builder lands in the next commit. Head to{" "}
        <a href="/library" className="underline">
          /library
        </a>{" "}
        for now.
      </p>
    </main>
  );
}
