import type { Metadata } from "next";
import { Big_Shoulders, Instrument_Sans, Martian_Mono } from "next/font/google";
import { Shell } from "@/components/Shell";
import "./globals.css";

const bigShoulders = Big_Shoulders({
  variable: "--font-big-shoulders",
  subsets: ["latin"],
  weight: ["600", "800", "900"],
});
const instrument = Instrument_Sans({ variable: "--font-instrument", subsets: ["latin"] });
const martian = Martian_Mono({ variable: "--font-martian", subsets: ["latin"], weight: ["400"] });

export const metadata: Metadata = {
  title: "Recruitment intelligence",
  description: "Squad gaps, market search and tactics for the configured club.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${bigShoulders.variable} ${instrument.variable} ${martian.variable} h-full antialiased`}
    >
      <body className="min-h-full">
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}
