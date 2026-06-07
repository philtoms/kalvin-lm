/**
 * Auto-tune context guard
 *
 * Monitors context usage during auto-tune sessions. When context exceeds a
 * configurable ceiling, automatically spawns a fresh session with a resume
 * prompt that points the agent at session-state.md for re-anchoring.
 *
 * Config (in .pi/settings.json or ~/.pi/agent/settings.json):
 *   {
 *     "autoTune": {
 *       "contextCeiling": 0.8
 *     }
 *   }
 *
 * CLI flag (overrides settings):
 *   pi --auto-tune-ceiling 0.6
 *
 * Manual trigger:
 *   /auto-tune-handoff
 *
 * Default ceiling is 0.8 (80% of context window).
 */

import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const DEFAULT_CEILING = 0.8;
const SETTINGS_KEY = "autoTune";

interface AutoTuneSettings {
	contextCeiling?: number;
}

function loadSettings(cwd: string): AutoTuneSettings {
	for (const dir of [join(cwd, ".pi"), join(process.env.HOME ?? "~", ".pi/agent")]) {
		const path = join(dir, "settings.json");
		if (!existsSync(path)) continue;
		try {
			const raw = JSON.parse(readFileSync(path, "utf8"));
			if (raw[SETTINGS_KEY]) return raw[SETTINGS_KEY];
		} catch {
			// ignore malformed settings
		}
	}
	return {};
}

function findActiveAutoTuneSession(cwd: string): string | null {
	const autoTuneDir = join(cwd, "auto-tune");
	if (!existsSync(autoTuneDir)) return null;

	try {
		for (const entry of readdirSync(autoTuneDir, { withFileTypes: true })) {
			if (!entry.isDirectory()) continue;
			const stateFile = join(autoTuneDir, entry.name, "session-state.md");
			if (existsSync(stateFile)) {
				return entry.name;
			}
		}
	} catch {
		// permission or other fs error
	}
	return null;
}

function resolveCeiling(cwd: string, flagValue: number): number {
	if (typeof flagValue === "number" && flagValue > 0 && flagValue < 1) {
		return flagValue;
	}
	const settings = loadSettings(cwd);
	if (
		typeof settings.contextCeiling === "number" &&
		settings.contextCeiling > 0 &&
		settings.contextCeiling < 1
	) {
		return settings.contextCeiling;
	}
	return DEFAULT_CEILING;
}

export default function (pi: ExtensionAPI) {
	let triggered = false;

	pi.registerFlag("auto-tune-ceiling", {
		description: "Context ceiling (0-1) for auto-tune session handoff",
		type: "number",
		default: -1,
	});

	pi.on("session_start", () => {
		triggered = false;
	});

	// Auto-detect: when context crosses the ceiling, queue the handoff command
	pi.on("turn_end", (_event, ctx) => {
		if (triggered) return;

		const usage = ctx.getContextUsage();
		if (!usage || usage.tokens === null || !usage.contextWindow) return;

		const sessionName = findActiveAutoTuneSession(ctx.cwd);
		if (!sessionName) return;

		const ceiling = resolveCeiling(ctx.cwd, pi.getFlag("auto-tune-ceiling") as number);
		const percent = usage.percent ?? ((usage.tokens / usage.contextWindow) * 100);

		if (percent < ceiling * 100) return;

		triggered = true;

		ctx.ui.notify(
			`Context at ${Math.round(percent)}% — ceiling ${Math.round(ceiling * 100)}%. ` +
				`Auto-spawning fresh session for auto-tune/${sessionName}...`,
			"warning",
		);

		// Queue the command as a follow-up so it runs after the current turn completes
		pi.sendUserMessage("/auto-tune-handoff", { deliverAs: "followUp" });
	});

	// The actual handoff logic — needs ExtensionCommandContext for newSession
	pi.registerCommand("auto-tune-handoff", {
		description: "Spawn a fresh session to continue auto-tune (reads session-state.md)",
		handler: async (_args, ctx) => {
			const sessionName = findActiveAutoTuneSession(ctx.cwd);
			if (!sessionName) {
				ctx.ui.notify("No active auto-tune session found (no session-state.md)", "error");
				return;
			}

			const statePath = `auto-tune/${sessionName}/session-state.md`;
			const currentSessionFile = ctx.sessionManager.getSessionFile();

			ctx.ui.notify(`Handing off to fresh session for auto-tune/${sessionName}...`, "info");

			const result = await ctx.newSession({
				parentSession: currentSessionFile ?? undefined,
				withSession: async (replacementCtx) => {
					await replacementCtx.sendUserMessage(
						`Resume auto-tune ${sessionName}. ` +
							`Your first action is to read ${statePath} to re-anchor. ` +
							`Then continue from the Current Phase and Next Action described there.`,
					);
				},
			});

			if (result.cancelled) {
				ctx.ui.notify("Handoff cancelled", "info");
			}
		},
	});
}
