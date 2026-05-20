"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type MixPlan, type MixPlanStatus } from "@/lib/api";

function statusBadgeClass(status: MixPlanStatus): string {
  switch (status) {
    case "ready":
      return "bg-emerald-500/30";
    case "rendering":
      return "bg-amber-500/30";
    case "failed":
      return "bg-red-500/30";
    default:
      return "bg-zinc-500/30";
  }
}

export function MixPlanDebug({
  mixPlanId,
  nextTitle,
}: {
  mixPlanId: string;
  nextTitle: string;
}) {
  const qc = useQueryClient();

  const planQ = useQuery({
    queryKey: ["mix-plan", mixPlanId],
    queryFn: () => api.getMixPlan(mixPlanId),
    retry: false,
    // Poll only while the worker is actively rendering. All other states
    // are terminal-ish (pending = waiting on user, ready / failed = done).
    refetchInterval: (q) => {
      const p = q.state.data;
      if (p && p.status === "rendering") return 1500;
      return false;
    },
  });

  const render = useMutation({
    mutationFn: () => api.triggerRenderMixPlan(mixPlanId),
    onMutate: () => {
      // Optimistic flip to "rendering" so the polling engages immediately,
      // matching the pattern used by separate/transcribe on the song page.
      qc.setQueryData<MixPlan | null | undefined>(
        ["mix-plan", mixPlanId],
        (prev) => (prev ? { ...prev, status: "rendering", error_text: null } : prev)
      );
    },
    onError: () => {
      qc.invalidateQueries({ queryKey: ["mix-plan", mixPlanId] });
    },
  });

  if (planQ.isLoading) {
    return (
      <section className="flex flex-col gap-3 border rounded p-4">
        <h2 className="font-semibold">Transition</h2>
        <p className="text-xs opacity-70">Loading mix plan…</p>
      </section>
    );
  }

  const plan = planQ.data;
  if (!plan) {
    return null;
  }

  const isRendering = plan.status === "rendering";
  const isReady = plan.status === "ready";
  const isFailed = plan.status === "failed";
  const hasAudio = isReady && !!plan.rendered_audio_path;
  const buttonBusy = render.isPending || isRendering;

  return (
    <section className="flex flex-col gap-3 border rounded p-4">
      <header className="flex items-center justify-between">
        <div className="flex flex-col gap-0.5 min-w-0">
          <h2 className="font-semibold">Transition</h2>
          <p className="text-xs opacity-70 truncate">→ {nextTitle}</p>
        </div>
        <span
          className={
            "text-xs px-2 py-1 rounded " + statusBadgeClass(plan.status)
          }
        >
          {plan.status}
        </span>
      </header>

      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => render.mutate()}
          disabled={buttonBusy}
          className="text-sm border rounded px-3 py-1 hover:bg-black/5 dark:hover:bg-white/10 disabled:opacity-50"
        >
          {buttonBusy
            ? "Rendering…"
            : hasAudio
              ? "Re-render transition"
              : "Render transition"}
        </button>
      </div>

      {hasAudio && (
        <audio
          controls
          src={api.mixPlanAudioUrl(plan.id)}
          className="w-full"
        />
      )}

      {isFailed && plan.error_text && (
        <pre className="text-xs text-red-700 dark:text-red-400 whitespace-pre-wrap break-words border border-red-500/30 rounded p-2 bg-red-500/5">
          {plan.error_text}
        </pre>
      )}

      <details className="text-xs">
        <summary className="cursor-pointer opacity-70 hover:opacity-100">
          Plan JSON
        </summary>
        <pre className="mt-2 max-h-72 overflow-auto border rounded p-2 bg-black/5 dark:bg-white/5 whitespace-pre-wrap break-words">
          {plan.plan_json
            ? JSON.stringify(plan.plan_json, null, 2)
            : "— (no plan yet)"}
        </pre>
      </details>
    </section>
  );
}
