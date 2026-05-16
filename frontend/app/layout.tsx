import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI DJ",
  description: "AI-mixed DJ sets from a YouTube queue",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
