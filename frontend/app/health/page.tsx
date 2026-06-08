type Health = {
  status: string;
  db: string;
  redis: string;
};

// Server-side only — resolves to http://backend:8000 inside docker compose,
// http://localhost:8000 when the frontend runs on the host.
const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";

async function fetchHealth(): Promise<Health | { error: string }> {
  try {
    const res = await fetch(`${BACKEND_URL}/health`, { cache: "no-store" });
    if (!res.ok) return { error: `Backend returned ${res.status}` };
    return (await res.json()) as Health;
  } catch (e) {
    return { error: e instanceof Error ? e.message : String(e) };
  }
}

export default async function HomePage() {
  const health = await fetchHealth();

  return (
    <main className="min-h-screen flex flex-col items-center justify-center gap-6 p-8 font-mono">
      <h1 className="text-3xl font-bold">MixMind</h1>
      <p className="text-sm opacity-70">System health</p>

      <section className="border rounded-lg p-6 min-w-[320px]">
        <h2 className="font-semibold mb-3">Backend health</h2>
        <pre className="text-sm bg-black/5 dark:bg-white/10 p-3 rounded">
          {JSON.stringify(health, null, 2)}
        </pre>
      </section>
    </main>
  );
}
