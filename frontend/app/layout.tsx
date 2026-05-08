import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { ThemeProvider } from "@/lib/theme";
import "./globals.css";

const geistSans = Geist({
	variable: "--font-geist-sans",
	subsets: ["latin"],
});

const geistMono = Geist_Mono({
	variable: "--font-geist-mono",
	subsets: ["latin"],
});

export const metadata: Metadata = {
	title: "SentimentFlow",
	description: "中文情感分析训练与推理平台",
};

export default function RootLayout({
	children,
}: Readonly<{
	children: React.ReactNode;
}>) {
	return (
		<html
			lang="zh-CN"
			suppressHydrationWarning
			className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
		>
			<body
				suppressHydrationWarning
				className="min-h-full flex flex-col"
			>
				<ThemeProvider>{children}</ThemeProvider>
			</body>
		</html>
	);
}
