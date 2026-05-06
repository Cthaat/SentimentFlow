"use client";

import { useSyncExternalStore } from "react";
import { Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useTheme } from "@/lib/theme";

function subscribeMounted(listener: () => void) {
	const timer = window.setTimeout(listener, 0);
	return () => window.clearTimeout(timer);
}

function getMountedSnapshot() {
	return true;
}

function getServerMountedSnapshot() {
	return false;
}

export function ThemeToggle() {
	const { theme, toggleTheme } = useTheme();
	const mounted = useSyncExternalStore(
		subscribeMounted,
		getMountedSnapshot,
		getServerMountedSnapshot,
	);

	const icon = !mounted ? (
		<span className="h-4 w-4" suppressHydrationWarning />
	) : theme === "light" ? (
		<Moon className="h-4 w-4" suppressHydrationWarning />
	) : (
		<Sun className="h-4 w-4" suppressHydrationWarning />
	);

	return (
		<Button
			variant="outline"
			size="icon"
			onClick={toggleTheme}
			aria-label="Toggle theme"
			suppressHydrationWarning
		>
			{icon}
		</Button>
	);
}
