"use client";

import { Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useTheme } from "@/lib/theme";

export function ThemeToggle() {
	const { theme, toggleTheme } = useTheme();

	return (
		<Button
			variant="outline"
			size="icon"
			onClick={toggleTheme}
			aria-label="Toggle theme"
		>
			{theme === "light" ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
		</Button>
	);
}
