"use client";

import React, { useEffect, useState } from "react";

interface ThemeContextType {
	theme: "light" | "dark";
	toggleTheme: () => void;
}

const ThemeContext = React.createContext<ThemeContextType | undefined>(undefined);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
	const [theme, setTheme] = useState<"light" | "dark" | null>(null);

	useEffect(() => {
		// 只在客户端执行
		const stored = localStorage.getItem("theme") as "light" | "dark" | null;
		const init = stored || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
		setTheme(init);
		applyTheme(init);
	}, []);

	const applyTheme = (newTheme: "light" | "dark") => {
		if (newTheme === "dark") {
			document.documentElement.classList.add("dark");
		} else {
			document.documentElement.classList.remove("dark");
		}
		localStorage.setItem("theme", newTheme);
	};

	const toggleTheme = () => {
		if (theme) {
			const newTheme = theme === "light" ? "dark" : "light";
			setTheme(newTheme);
			applyTheme(newTheme);
		}
	};

	// 在主题加载前渲染子组件，避免水合错误
	return <ThemeContext.Provider value={{ theme: theme || "light", toggleTheme }}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
	const context = React.useContext(ThemeContext);
	if (!context) {
		throw new Error("useTheme must be used within ThemeProvider");
	}
	return context;
}
