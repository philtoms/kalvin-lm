/**
 * Auto-tune context guard
 *
 * Monitors context usage during auto-tune sessions. When context exceeds a
 * configurable ceiling, injects a steering message telling the LLM to wrap up,
 * then automatically creates a fresh session to continue.
 *
 * Flow:
 *   1. User runs /auto-tune → extension caches ctx.newSession from
 *      ExtensionCommandContext, then forwards to the skill system
 *   2. Skill runs normally, auto-tune loop begins
 *   3. Context crosses ceiling → turn_end handler injects steering message,
 *      then starts polling ctx.isIdle() via setTimeout
 *   4. Agent finishes responding to steering message → goes idle
 *   5. Polling detects idle → calls cached newSession() from a macrotask
 *      (setTimeout), avoiding the deadlock that occurs when calling
 *      session-control functions from event handlers or microtasks
 *
 * Manual handoff is also available via /auto-tune-handoff.
 *
 * Why setTimeout polling instead of waitForIdle()?
 *   waitForIdle() returns activeRun.promise which resolves when the agent loop
 *   finishes. The .then() callback runs as a microtask that can interleave with
 *   _agentEventQueue processing. Calling newSession() from that microtask
 *   triggers emit(session_before_switch) which deadlocks because the extension
 *   runner's emit system is already busy. setTimeout callbacks are macrotasks
 *   that run after all microtask/event processing completes, so the emit system
 *   is free.
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
 *   /auto-tune          Start/resume auto-tune (caches newSession for auto-handoff)
 *   /auto-tune-handoff  Handoff to fresh session (manual or auto-triggered)
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

interface SessionInfo {
	name: string;
	/** Path to session-state.md relative to the main repo root (for resume messages). */
	statePath: string;
	/** Whether the session lives in a worktree. */
	inWorktree: boolean;
}

function findActiveAutoTuneSession(cwd: string): SessionInfo | null {
	// 1. Check worktrees first — active sessions typically run in worktrees.
	const worktreesDir = join(cwd, ".worktrees", "auto-tune");
	if (existsSync(worktreesDir)) {
		try {
			for (const entry of readdirSync(worktreesDir, { withFileTypes: true })) {
				if (!entry.isDirectory()) continue;
				const stateFile = join(
					worktreesDir, entry.name, "auto-tune", entry.name, "session-state.md",
				);
				if (existsSync(stateFile)) {
					return {
						name: entry.name,
						statePath: `.worktrees/auto-tune/${entry.name}/auto-tune/${entry.name}/session-state.md`,
						inWorktree: true,
					};
				}
			}
		} catch {
			// permission or other fs error
		}
	}

	// 2. Check auto-tune/ directly (main repo or inside a worktree)
	const autoTuneDir = join(cwd, "auto-tune");
	if (existsSync(autoTuneDir)) {
		try {
			for (const entry of readdirSync(autoTuneDir, { withFileTypes: true })) {
				if (!entry.isDirectory()) continue;
				const stateFile = join(autoTuneDir, entry.name, "session-state.md");
				if (existsSync(stateFile)) {
					return {
						name: entry.name,
						statePath: `auto-tune/${entry.name}/session-state.md`,
						inWorktree: false,
					};
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

	// Cached newSession from ExtensionCommandContext.
	// Populated by /auto-tune on every invocation. Only newSession is cached
	// (not waitForIdle) — idle detection uses ctx.isIdle() polling instead.
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	let cachedNewSession: ((...args: any[]) => Promise<any>) | null = null;

	// Cached session info from /auto-tune invocation. Used during handoff to
	// resume the correct session instead of guessing from the filesystem.
	let cachedSessionName: string | null = null;
	let cachedStatePath: string | null = null;
	let cachedInWorktree: boolean = false;

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

		// Resolve session info — use cache (set by /auto-tune) when available,
		// otherwise discover from filesystem.
		const session = cachedSessionName
			? { name: cachedSessionName, statePath: cachedStatePath ?? `auto-tune/${cachedSessionName}/session-state.md`, inWorktree: cachedInWorktree }
			: findActiveAutoTuneSession(ctx.cwd);
		if (!session) {
			console.error("[auto-tune-handoff] No active auto-tune session found in", ctx.cwd);
			return;
		}
		const { name: sessionName } = session;

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

		if (cachedNewSession) {
			// Automatic handoff — poll for idle via setTimeout, then call newSession
			// from a macrotask to avoid the deadlock that occurs when calling
			// session-control functions from event handlers or microtasks.
			autoHandoffPending = true;

			ctx.ui.notify(
				`Context at ${Math.round(percent)}% — ceiling ${Math.round(ceiling * 100)}%. ` +
					`Auto-handoff to fresh session...`,
				"warning",
			);

			const pollForIdleAndHandoff = () => {
				if (!autoHandoffPending || !cachedNewSession) return;

				if (!ctx.isIdle()) {
					setTimeout(pollForIdleAndHandoff, 200);
					return;
				}

				// Agent is idle. Call cachedNewSession from this setTimeout macrotask.
				autoHandoffPending = false;

				const currentSessionFile = ctx.sessionManager.getSessionFile();

				console.error(
					`[auto-tune-handoff] Agent idle — auto-handoff for session ${sessionName}`,
				);

				cachedNewSession!({
					parentSession: currentSessionFile ?? undefined,
					withSession: async (replacementCtx: any) => {
						let resumeMsg =
							`Resume auto-tune ${sessionName}. ` +
							`Your first action is to read ${session.statePath} to re-anchor. ` +
							`Then continue from the Current Phase and Next Action described there.`;
						if (session.inWorktree) {
							resumeMsg += ` First cd into .worktrees/auto-tune/${sessionName}/`;
						}
						await replacementCtx.sendUserMessage(resumeMsg);
					},
				})
					.then((result: any) => {
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
			};

			// Start polling after a short delay to give the steering message time
			// to be picked up by the agent loop.
			setTimeout(pollForIdleAndHandoff, 500);
		} else {
			// Fallback: no cached newSession. Ask user to run the command.
			ctx.ui.notify(
				`Context at ${Math.round(percent)}% — ceiling ${Math.round(ceiling * 100)}%. ` +
					`Run /auto-tune-handoff to continue in a fresh session.`,
				"warning",
			);
		}
	});

	// ── /auto-tune — skill entry point + newSession cache ─────────────

	pi.registerCommand("auto-tune", {
		description: "Start or resume an auto-tune session (caches newSession for auto-handoff)",
		handler: async (args, ctx) => {
			// Cache newSession from this command context for automatic handoff.
			// Only newSession is cached — idle detection uses ctx.isIdle() polling
			// via setTimeout to avoid the deadlock from calling session-control
			// functions from event handlers or microtasks.
			cachedNewSession = ctx.newSession.bind(ctx);

			// Cache session info so handoff resumes the correct session.
			const trimmed = args.trim();
			if (trimmed) {
				cachedSessionName = trimmed.split(/\s+/)[0];
				// Resolve state path: check if this session lives in a worktree.
				const discovered = findActiveAutoTuneSession(ctx.cwd);
				if (discovered && discovered.name === cachedSessionName) {
					cachedStatePath = discovered.statePath;
					cachedInWorktree = discovered.inWorktree;
				} else {
					cachedStatePath = `auto-tune/${cachedSessionName}/session-state.md`;
					cachedInWorktree = false;
				}
			}

			// Forward to pi's skill system. sendUserMessage does NOT re-trigger
			// command handling, so the LLM receives the text and loads the skill
			// based on its system prompt skill descriptions.
			const prompt = trimmed
				? `/auto-tune ${trimmed}`
				: "/auto-tune";

			pi.sendUserMessage(prompt);
		},
	});

	// ── /auto-tune-handoff — manual handoff ───────────────────────────

	pi.registerCommand("auto-tune-handoff", {
		description: "Spawn a fresh session to continue auto-tune (reads session-state.md)",
		handler: async (_args, ctx) => {
			// Cancel any pending auto-handoff (user is doing it manually)
			autoHandoffPending = false;

			// Resolve session info — use cache when available, otherwise discover.
			const session = cachedSessionName
				? { name: cachedSessionName, statePath: cachedStatePath ?? `auto-tune/${cachedSessionName}/session-state.md`, inWorktree: cachedInWorktree }
				: findActiveAutoTuneSession(ctx.cwd);
			if (!session) {
				ctx.ui.notify(
					"No active auto-tune session found (no session-state.md)",
					"error",
				);
				return;
			}
			const { name: sessionName, statePath, inWorktree } = session;
			const currentSessionFile = ctx.sessionManager.getSessionFile();

			ctx.ui.notify(
				`Handing off to fresh session for auto-tune/${sessionName}...`,
				"info",
			);

			const result = await ctx.newSession({
				parentSession: currentSessionFile ?? undefined,
				withSession: async (replacementCtx) => {
					let resumeMsg =
						`Resume auto-tune ${sessionName}. ` +
						`Your first action is to read ${statePath} to re-anchor. ` +
						`Then continue from the Current Phase and Next Action described there.`;
					if (inWorktree) {
						resumeMsg += ` First cd into .worktrees/auto-tune/${sessionName}/`;
					}
					await replacementCtx.sendUserMessage(resumeMsg);
				},
			});

			if (result.cancelled) {
				ctx.ui.notify("Handoff cancelled", "info");
			}
		},
	});
}
