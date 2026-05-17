"use client";

import { useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import type { QueueItem } from "@/lib/api";

type Props = {
  item: QueueItem;
  onRemove: (itemId: string) => void;
};

function statusBadgeClass(status: string): string {
  if (status === "failed") return "bg-red-500/20";
  if (status === "analyzed" || status === "ready") return "bg-green-500/20";
  if (status === "downloaded") return "bg-blue-500/20";
  return "bg-yellow-500/20";
}

export function QueueItemRow({ item, onRemove }: Props) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } =
    useSortable({ id: item.id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.4 : 1,
  };

  const s = item.song;
  return (
    <li
      ref={setNodeRef}
      style={style}
      className="flex items-center gap-3 border rounded p-2 bg-background"
    >
      <button
        type="button"
        {...attributes}
        {...listeners}
        className="cursor-grab active:cursor-grabbing text-lg opacity-60 px-1"
        aria-label="Drag to reorder"
      >
        ⋮⋮
      </button>
      <span className="text-sm opacity-60 w-6 tabular-nums">
        {item.position + 1}
      </span>
      {s.thumbnail_url && (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={s.thumbnail_url}
          alt=""
          className="w-16 h-10 object-cover rounded"
        />
      )}
      <div className="flex-1 min-w-0">
        <p className="font-medium truncate">{s.title}</p>
        <p className="text-xs opacity-70 truncate">{s.artist ?? "—"}</p>
      </div>
      <span className={`text-xs px-2 py-1 rounded ${statusBadgeClass(s.status)}`}>
        {s.status}
      </span>
      <button
        type="button"
        onClick={() => onRemove(item.id)}
        className="text-sm opacity-60 hover:opacity-100 px-2"
        aria-label="Remove from queue"
      >
        ×
      </button>
    </li>
  );
}
