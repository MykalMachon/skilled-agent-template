import { anthropic } from '@ai-sdk/anthropic';
import { ollama } from 'ollama-ai-provider-v2';
import { Experimental_Agent as Agent, stepCountIs } from 'ai';
import type { AssistantModelMessage, UserModelMessage } from 'ai';
import matter from 'gray-matter';
import { z } from 'zod';

import { readdir } from 'node:fs/promises';

const SKILLS_PATH = `${import.meta.dir}/skills`;

// ? MODEL PROVIDER CONFIGURATION
const MODEL_PROVIDER = (Bun.env.MODEL_PROVIDER || 'anthropic') as 'anthropic' | 'ollama';
const MODEL_NAME = Bun.env.MODEL_NAME || (MODEL_PROVIDER === 'anthropic' ? 'claude-haiku-4-5' : 'ministral-3:8b');

// Create the model instance based on the provider
const getModel = () => {
	switch (MODEL_PROVIDER) {
		case 'anthropic':
			return anthropic(MODEL_NAME);
		case 'ollama':
			return ollama(MODEL_NAME);
		default:
			throw new Error(`Unknown model provider: ${MODEL_PROVIDER}`);
	}
};


// ? SYSTEM PROMPT CREATION
// useful for progressive discovery pattern in skill-using agents

interface SkillPreview {
	name: string;
	description: string;
	allowedTools?: string[];
	license?: string;
	compatibility?: string;
	metadata?: Record<string, any>;
}

const buildSkillsListMd = async () => {
	const lines = [];

	// parse skills.md files using marked
	const filePathsInSkillsFolder = await readdir(SKILLS_PATH, { recursive: true });

	// ensure that the file ends in SKILLS.md and that it's only one level deep
	const skillFilePaths = filePathsInSkillsFolder.filter((path) => path.endsWith('SKILL.md') && path.split('/').length == 2)

	// parse the skills files 
	for (const skillFilePath of skillFilePaths) {
		const skillFile = Bun.file(`${SKILLS_PATH}/${skillFilePath}`);
		const skillFileText = await skillFile.text();
		const { data } = matter(skillFileText) as unknown as { data: SkillPreview };
		lines.push(`- **${data.name}**: ${data.description} (${SKILLS_PATH}/${skillFilePath})`);
	}

	// return the joined lines
	if (lines.length == 0) return ''
	return lines.join('\n');
}

const buildPrompt = async () => {
	const skillsList = await buildSkillsListMd();
	return `
# Skilled Agent

You are a skilled agent that can learn using skills that are just a bunch of markdown and small programs.

## Tone and formatting

You are being used for serious work, so try and be both serious and concise in your responses.
Use only simple markdown. currently your output is displayed in a terminal ui.

## Skills Usage Pattern

1. **Discover**: When you receive a user request, first think about which skills might be relevant
2. **Read**: Always read the full SKILL.md document before using a skill
3. **Execute**: Use the tools provided by the skill to accomplish the task
4. **Verify**: Check the results and provide feedback to the user

## Available Skills

${skillsList || 'No skills available currently.'}

## Important Rules

- ALWAYS read the SKILL.md document before using any skill-related tools
- If a skill provides scripts, they are located in the \`scripts/\` subdirectory
- Be explicit about what you're doing and why when using skills
- Follow the skill usage pattern for best results
`;
}


// ? TOOL DEFINITIONS

// * Tool: runScript
const runScriptTool = {
	description: 'Run an executable script from the skills directory',
	inputSchema: z.object({
		scriptPath: z.string().refine(path => {
			// Normalize path and ensure it's within SKILLS_PATH
			const normalized = path.replace(/\\/g, '/');
			return normalized.startsWith(SKILLS_PATH) &&
				!normalized.includes('../') &&
				normalized.split('/').length >= 3; // Ensure it's at least skills/<skill>/<folder>/
		}, {
			message: "Script path must be within the skills directory"
		}),
		args: z.array(z.string()).optional()
	}),
	execute: async ({ scriptPath, args = [] }: { scriptPath: string; args?: string[] }) => {
		try {
			const file = Bun.file(scriptPath);
			if (!await file.exists()) {
				return 'Script not found';
			}

			// Check if file is executable
			try {
				await Bun.$`test -x ${scriptPath}`;
			} catch {
				return 'Script is not executable';
			}

			const proc = Bun.spawn([scriptPath, ...args], {
				timeout: 10000, // 10 second timeout
				stderr: 'pipe'
			});

			const [stdout, stderr] = await Promise.all([
				proc.stdout.text(),
				proc.stderr.text()
			]);

			await proc.exited;

			if (proc.exitCode !== 0) {
				return `Script failed with exit code ${proc.exitCode}: ${stderr}`;
			}

			return stdout;
		} catch (error) {
			return `Error executing script: ${error instanceof Error ? error.message : String(error)}`;
		}
	}
};

// * Tool: listDirectory
const listDirectoryTool = {
	description: 'List all files and directories in a given path',
	inputSchema: z.object({
		dirPath: z.string()
	}),
	execute: async ({ dirPath }: { dirPath: string }) => {
		try {
			const entries = await readdir(dirPath, { withFileTypes: true });
			return entries
				.map(entry => entry.isDirectory() ? `${entry.name}/` : entry.name)
				.join('\n');
		} catch (err) {
			if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
				return 'Directory not found';
			}
			throw err;
		}
	}
};

// * Tool: readFile
const readFileTool = {
	description: 'Read the contents of a file at the given path',
	inputSchema: z.object({
		filePath: z.string().refine(path => {
			const normalized = path.replace(/\\/g, '/');
			// Prevent reading sensitive files
			return !normalized.includes('/.env') &&
				!normalized.includes('/.git/') &&
				!normalized.includes('node_modules/') &&
				!normalized.match(/\.(png|jpg|jpeg|gif|bmp|webp|avif|pdf)$/i);
		}, {
			message: "Cannot read that file path"
		})
	}),
	execute: async ({ filePath }: { filePath: string }) => {
		try {
			const file = Bun.file(filePath);
			if (!await file.exists()) {
				return 'File not found';
			}

			// Limit file size to prevent large file reads
			const size = (file.size);
			if (size > 1000000) { // 1MB limit
				return 'File too large to read';
			}

			return await file.text();
		} catch (error) {
			return `Error reading file: ${error instanceof Error ? error.message : String(error)}`;
		}
	}
};


// ? TERMINAL STYLING
const colors = {
	reset: '\x1b[0m',
	bold: '\x1b[1m',
	dim: '\x1b[2m',

	// Text colors
	cyan: '\x1b[36m',
	green: '\x1b[32m',
	yellow: '\x1b[33m',
	magenta: '\x1b[35m',
	blue: '\x1b[34m',
	gray: '\x1b[90m',

	// Background colors
	bgCyan: '\x1b[46m',
	bgGreen: '\x1b[42m',
	bgYellow: '\x1b[43m',
	bgMagenta: '\x1b[45m',
};

const formatLabel = (label: string, color: string, bgColor?: string) => {
	const bg = bgColor || '';
	return `${bg}${color}${colors.bold}${label}${colors.reset}`;
};

const userLabel = formatLabel('User:', colors.cyan);
const agentLabel = formatLabel('Agent:', colors.green);
const toolCallLabel = (name: string) => `${colors.yellow}${colors.bold}⚡ Tool Call: ${name}${colors.reset}`;
const toolResultLabel = (name: string) => `${colors.magenta}${colors.bold}✓ Tool Result: ${name}${colors.reset}`;

// ? STREAMING HELPER
const compact = (obj: any, maxLen = 80): string => {
	const str = typeof obj === 'string' ? obj : JSON.stringify(obj);
	const oneLine = str.replace(/\s+/g, ' ');
	return oneLine.length > maxLen ? oneLine.slice(0, maxLen) + '...' : oneLine;
};

// Track conversation history
const conversationHistory: Array<UserModelMessage | AssistantModelMessage> = [];

async function streamAgentResponse(prompt: string) {
	// Add user message to history
	conversationHistory.push({
		role: 'user',
		content: prompt
	});

	const { textStream, fullStream } = agent.stream({
		messages: conversationHistory
	});

	// Display tool calls and results as they happen
	const toolMonitor = (async () => {
		for await (const chunk of fullStream) {
			switch (chunk.type) {
				case 'tool-call':
					console.log(`\n${toolCallLabel(chunk.toolName)} ${colors.dim}${compact(chunk.input)}${colors.reset}`);
					break;
				case 'tool-result':
					console.log(`${toolResultLabel(chunk.toolName)} ${colors.dim}${compact(chunk.output)}${colors.reset}`);
					break;
			}
		}
	})();

	// Stream and display the text output
	let assistantResponse = '';
	process.stdout.write(`${agentLabel} `);

	for await (const textPart of textStream) {
		process.stdout.write(textPart);
		assistantResponse += textPart;
	}

	console.log('\n');

	await toolMonitor;

	// Add assistant response to history
	conversationHistory.push({
		role: 'assistant',
		content: assistantResponse
	});
}

// * init agent
const agent = new Agent({
	model: getModel(),
	system: await buildPrompt(),
	prepareStep: async ({ messages }) => {
		// Keep only recent messages to stay within context limits
		if (messages.length > 8) {
			const filteredMessages = [
				messages[0], // Keep system message
				...messages.slice(-8), // Keep last 8 messages
				// vvv shoutout the internet for this one vvv
			].filter((msg): msg is NonNullable<typeof msg> => msg !== undefined);
			return {
				messages: filteredMessages,
			};
		}
		return {};
	},
	tools: {
		runScript: runScriptTool,
		listDirectory: listDirectoryTool,
		readFile: readFileTool
	},
	stopWhen: stepCountIs(10)
});

// ? INTERACTIVE CHAT LOOP
console.log(`${colors.bold}${colors.blue}╔════════════════════════════════════════╗${colors.reset}`);
console.log(`${colors.bold}${colors.blue}║    Skilled Agent - Starter Template    ║${colors.reset}`);
console.log(`${colors.bold}${colors.blue}╚════════════════════════════════════════╝${colors.reset}\n`);
console.log(`${colors.dim}Type your message and press Enter. Press Ctrl+C to exit.${colors.reset}\n`);

// Start the conversation loop using Bun's console iterator
for await (const line of console) {
	if (!line.trim()) {
		process.stdout.write(`${userLabel} `);
		continue;
	}

	await streamAgentResponse(line);

	// Prompt for next input
	process.stdout.write(`${userLabel} `);
}
