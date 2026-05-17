import Link from "next/link";

type Props = {
  subtitle?: string;
};

export function NavHeader({ subtitle }: Props) {
  return (
    <header className="flex items-baseline justify-between">
      <h1 className="text-3xl font-bold">
        <Link href="/">AI DJ</Link>
      </h1>
      <nav className="flex items-baseline gap-4 text-sm">
        <Link href="/" className="hover:underline">
          Queue
        </Link>
        <Link href="/library" className="hover:underline">
          Library
        </Link>
        {subtitle && <span className="opacity-70 text-xs">{subtitle}</span>}
      </nav>
    </header>
  );
}
