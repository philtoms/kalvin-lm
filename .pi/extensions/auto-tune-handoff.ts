/**
 * Auto-tune context guard
 *
 * Monitors context usage during auto-tune sessions. When context exceeds a
 * configurable ceiling, injects a steering message telling the LLM to wrap up,
 * then automatically creates a fresh session to continue.
 *
 * Bootstrapping: the /auto-tune command caches newSession/waitForIdle from the
 * ExtensionCommandContext on every invocation, then forwards to the skill system
 * via pi.sendUserMessage. This gives the extension the session-control functions
 * it needs for automatic handoff — no separate bootstrap step required.
 *
 * Flow:
 *   1. User runs /auto-tune → extension caches ctx.newSession/waitForIdle,
 *      then forwards to skill via pi.sendUserMessage
 *   2. Skill runs normally, auto-tune loop begins
 *   3. Context crosses ceiling → extension steers LLM to wrap up,
 *      waits for idle, then calls cachedNewSession() automatically
 *
 * Manual handoff is also available via /auto-tune-handoff.
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
 * Commands:
 *   /auto-tune          Start/resume auto-tune (caches handoff functions)
 *   /auto-tune-handoff  Manual handoff to fresh session
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
				const stateFile = join(
					worktreesDir, entry.name, "auto-tune", entry.name, "session-state.md",
				);
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
	let autoHandoffPending = false;

	// Cached command-context functions for automatic handoff.
	// Populated by /auto-tune (on every invocation) and /auto-tune-handoff.
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	let cachedNewSession: ((...args: any[]) => Promise<any>) | null = null;
	let cachedWaitForIdle: (() => Promise<void>) | null = null;

	function cacheCommandContext(
		newSession: typeof cachedNewSession,
		waitForIdle: typeof cachedWaitForIdle,
	) {
		cachedNewSession = newSession;
		cachedWaitForIdle = waitForIdle;
		console.error("[auto-tune-handoff] Cached newSession/waitForIdle for automatic handoff");
	}

	pi.registerFlag("auto-tune-ceiling", {
		description: "Context ceiling (0-1) for auto-tune session handoff",
		type: "number",
		default: -1,
	});

	pi.on("session_start", () => {
		triggered = false;
		autoHandoffPending = false;
	});

	// ── Context ceiling guard ─────────────────────────────────────────

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
			`percent=${percent.toFixed(1)}% ceiling=${(ceiling * 100).toFixed(0)}% ` +
			`flag=${flagVal} session=${sessionName} cached=${!!cachedNewSession}`,
		);

		if (percent < ceiling * 100) return;

		triggered = true;

		const statePath = `auto-tune/${sessionName}/session-state.md`;

		// Inject a steering message so the LLM wraps up before handoff.
		pi.sendMessage(
			{
				customType: "auto-tune-handoff",
				content:
					`[auto-tune-handoff] Context usage has reached ${Math.round(percent)}% ` +
					`(ceiling ${Math.round(ceiling * 100)}%). ` +
					`You MUST stop your current work immediately. ` +
					`Update auto-tune/${sessionName}/session-state.md with your latest observations ` +
					`(Current Phase, Next Action, latest run summary). ` +
					`Then tell the user: "Context is at ${Math.round(percent)}%. ` +
					`Auto-handoff starting in a moment — no action needed." ` +
					`Do NOT continue the auto-tune loop.`,
				display: true,
			},
			{ deliverAs: "steer" },
		);

		if (cachedNewSession && cachedWaitForIdle) {
			// Automatic handoff — wait for agent to finish, then create fresh session.
			autoHandoffPending = true;

			ctx.ui.notify(
				`Context at ${Math.round(percent)}% — ceiling ${Math.round(ceiling * 100)}%. ` +
					`Auto-handoff to fresh session...`,
				"warning",
			);

			cachedWaitForIdle()
				.then(() => {
					if (!autoHandoffPending || !cachedNewSession) return;
					autoHandoffPending = false;

					const currentSessionFile = ctx.sessionManager.getSessionFile();

					console.error(
						`[auto-tune-handoff] Agent idle — auto-handoff for session ${sessionName}`,
					);
					return cachedNewSession!({
						parentSession: currentSessionFile ?? undefined,
						withSession: async (replacementCtx) => {
							await replacementCtx.sendUserMessage(
								`Resume auto-tune ${sessionName}. ` +
									`Your first action is to read ${statePath} to re-anchor. ` +
									`Then continue from the Current Phase and Next Action described there.`,
							);
						},
					});
				})
				.then((result) => {
					if (result?.cancelled) {
						console.error("[auto-tune-handoff] Auto-handoff cancelled");
					}
				})
				.catch((err: Error) => {
					console.error("[auto-tune-handoff] Auto-handoff failed:", err);
					ctx.ui.notify(
						"Auto-handoff failed. Run /auto-tune-handoff manually.",
						"error",
					);
				});
		} else {
			// Fallback: no cached functions. Ask user to run /auto-tune-handoff.
			ctx.ui.notify(
				`Context at ${Math.round(percent)}% — ceiling ${Math.round(ceiling * 100)}%. ` +
					`Run /auto-tune-handoff to continue in a fresh session.`,
				"warning",
			);
		}
	});

	// ── /auto-tune — skill entry point + context cache ────────────────

	pi.registerCommand("auto-tune", {
		description: "Start or resume an auto-tune session (caches handoff context)",
		handler: async (args, ctx) => {
			// Cache the command-context functions for automatic handoff.
			// This happens on every /auto-tune invocation, not just the first.
			cacheCommandContext(ctx.newSession.bind(ctx), ctx.waitForIdle.bind(ctx));

			// Forward to pi's skill system. sendUserMessage does NOT re-trigger
			// command handling, so the LLM receives the text and loads the skill
			// based on its system prompt skill descriptions.
			const prompt = args.trim()
				? `/auto-tune ${args.trim()}`
				: "/auto-tune";

			pi.sendUserMessage(prompt);
		},
	});

	// ── /auto-tune-handoff — manual handoff ───────────────────────────

	pi.registerCommand("auto-tune-handoff", {
		description: "Spawn a fresh session to continue auto-tune (reads session-state.md)",
		handler: async (_args, ctx) => {
			cacheCommandContext(ctx.newSession.bind(ctx), ctx.waitForIdle.bind(ctx));

			// Cancel any pending auto-handoff (user is doing it manually)
			autoHandoffPending = false;

			const sessionName = findActiveAutoTuneSession(ctx.cwd);
			if (!sessionName) {
				ctx.ui.notify(
					"No active auto-tune session found (no session-state.md)",
					"error",
				);
				return;
			}

			const statePath = `auto-tune/${sessionName}/session-state.md`;
			const currentSessionFile = ctx.sessionManager.getSessionFile();

			ctx.ui.notify(
				`Handing off to fresh session for auto-tune/${sessionName}...`,
				"info",
			);

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
