/**
 * Auto-tune context guard
 *
 * Monitors context usage during auto-tune sessions. When context exceeds a
 * configurable ceiling, injects a steering message telling the LLM to wrap up,
 * and shows a notification prompting the user to run /auto-tune-handoff for a
 * fresh session.
 *
 * The handoff itself cannot be triggered automatically from an event handler
 * (pi's sendUserMessage skips extension command handling), so the extension
 * detects the condition and guides both the LLM and user to act.
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
	// 1. Check auto-tune/ directly (main repo or inside a worktree)
	const autoTuneDir = join(cwd, "auto-tune");
	if (existsSync(autoTuneDir)) {
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
	}

	// 2. Check worktrees: .worktrees/auto-tune/<name>/auto-tune/<name>/session-state.md
	const worktreesDir = join(cwd, ".worktrees", "auto-tune");
	if (existsSync(worktreesDir)) {
		try {
			for (const entry of readdirSync(worktreesDir, { withFileTypes: true })) {
				if (!entry.isDirectory()) continue;
				const stateFile = join(worktreesDir, entry.name, "auto-tune", entry.name, "session-state.md");
				if (existsSync(stateFile)) {
					return entry.name;
				}
			}
		} catch {
			// permission or other fs error
		}
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

	// Auto-detect: when context crosses the ceiling, steer the LLM to wrap up and prompt the user
	pi.on("turn_end", (_event, ctx) => {
		if (triggered) {
			console.error("[auto-tune-handoff] Already triggered, skipping");
			return;
		}

		const usage = ctx.getContextUsage();
		if (!usage || usage.tokens === null || !usage.contextWindow) {
			console.error("[auto-tune-handoff] No usage data:", JSON.stringify(usage));
			return;
		}

		const sessionName = findActiveAutoTuneSession(ctx.cwd);
		if (!sessionName) {
			console.error("[auto-tune-handoff] No active auto-tune session found in", ctx.cwd);
			return;
		}

		const flagVal = pi.getFlag("auto-tune-ceiling");
		const ceiling = resolveCeiling(ctx.cwd, flagVal as number);
		const percent = usage.percent ?? ((usage.tokens / usage.contextWindow) * 100);

		console.error(
			`[auto-tune-handoff] check: tokens=${usage.tokens} window=${usage.contextWindow} ` +
			`percent=${percent.toFixed(1)}% ceiling=${(ceiling * 100).toFixed(0)}% flag=${flagVal} session=${sessionName}`,
		);

		if (percent < ceiling * 100) return;

		triggered = true;

		ctx.ui.notify(
			`Context at ${Math.round(percent)}% — ceiling ${Math.round(ceiling * 100)}%. ` +
				`Run /auto-tune-handoff to continue in a fresh session.`,
			"warning",
		);

		// Inject a steering message so the LLM sees it before the next turn.
		// We cannot use sendUserMessage to trigger the command — it calls prompt()
		// with expandPromptTemplates: false which skips extension command handling.
		// sendMessage injects a custom message that convertToLlm() converts to a
		// user-visible message for the LLM.
		pi.sendMessage(
			{
				customType: "auto-tune-handoff",
				content:
					`[auto-tune-handoff] Context usage has reached ${Math.round(percent)}% ` +
					`(ceiling ${Math.round(ceiling * 100)}%). ` +
					`You MUST stop your current work immediately. ` +
					`Update auto-tune/${sessionName}/session-state.md with your latest observations (Current Phase, Next Action, latest run summary). ` +
					`Then tell the user: "Context is at ${Math.round(percent)}%. Run /auto-tune-handoff to continue in a fresh session." ` +
					`Do NOT continue the auto-tune loop.`,
				display: true,
			},
			{ deliverAs: "steer" },
		);
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
