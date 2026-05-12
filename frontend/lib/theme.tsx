"use client";

import React, { useSyncExternalStore } from "react";

interface ThemeContextType {
	theme: "light" | "dark";
	toggleTheme: () => void;
}

const ThemeContext = React.createContext<ThemeContextType | undefined>(undefined);

type Theme = "light" | "dark";

const listeners = new Set<() => void>();
let currentTheme: Theme = "light";
let initialized = false;

function readStoredTheme(): Theme {
	if (typeof window === "undefined") {
		return "light";
	}

	try {
		const stored = window.localStorage.getItem("theme");
		if (stored === "light" || stored === "dark") {
			return stored;
		}
	} catch {
		// 浏览器禁用存储时使用系统主题或默认亮色。
	}

	try {
		return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
	} catch {
		return "light";
	}
}

function applyTheme(newTheme: Theme) {
	if (typeof document === "undefined") {
		return;
	}

	if (newTheme === "dark") {
		document.documentElement.classList.add("dark");
	} else {
		document.documentElement.classList.remove("dark");
	}
	try {
		window.localStorage.setItem("theme", newTheme);
	} catch {
		// 存储失败不影响当前页面主题切换。
	}
}

function emitThemeChange() {
	for (const listener of listeners) {
		listener();
	}
}

function subscribeTheme(listener: () => void) {
	listeners.add(listener);
	if (!initialized && typeof window !== "undefined") {
		initialized = true;
		currentTheme = readStoredTheme();
		applyTheme(currentTheme);
		queueMicrotask(listener);
	}

	return () => {
		listeners.delete(listener);
	};
}

function getThemeSnapshot(): Theme {
	return currentTheme;
}

function getServerThemeSnapshot(): Theme {
	return "light";
}

function setStoredTheme(newTheme: Theme) {
	currentTheme = newTheme;
	initialized = true;
	applyTheme(newTheme);
	emitThemeChange();
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
	const theme = useSyncExternalStore(subscribeTheme, getThemeSnapshot, getServerThemeSnapshot);

	const toggleTheme = () => {
		setStoredTheme(theme === "light" ? "dark" : "light");
	};

	return <ThemeContext.Provider value={{ theme, toggleTheme }}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
	const context = React.useContext(ThemeContext);
	if (!context) {
		throw new Error("useTheme must be used within ThemeProvider");
	}
	return context;
}
