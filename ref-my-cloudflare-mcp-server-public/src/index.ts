import { McpAgent } from "agents/mcp";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";

function getSymbolFromName(name: string): string {
    if (["bitcoin", "btc"].includes(name.toLowerCase())) return "BTCUSDT";
    if (["ethereum", "eth"].includes(name.toLowerCase())) return "ETHUSDT";
    // TypeScript's ternary operator - like Python's `x if condition else y`
    // The explicit type check is needed because TypeScript is more strict about types
    return typeof name === "string" ? name.toUpperCase() : String(name).toUpperCase();
}

// Define our MCP agent with tools
export class MyMCP extends McpAgent {
	server = new McpServer({
		name: "Crypto Price MCP",
		version: "1.0.0",
	});

	async init() {
		// Simple addition tool
		this.server.tool(
			"add",
			{ a: z.number(), b: z.number() },
			async ({ a, b }) => ({
				content: [{ type: "text", text: String(a + b) }],
			})
		);
		
		this.server.tool(
			// Tool name
			"get_price",
			// Parameter schema using zod (similar to Python's type hints but with runtime validation)
			{ symbol: z.string() },
			// Async handler function (similar to Python's async def)
			// Arrow function syntax is common in TypeScript: ({params}) => { body }
			async ({ symbol }) => {
				const resolvedSymbol = getSymbolFromName(symbol);
				const url = `https://mcp-course.s3.eu-central-1.amazonaws.com/public/hard-coded-price.json`;
				// fetch is built into modern Node.js (similar to Python's requests.get)
				const response = await fetch(url);
				if (!response.ok) {
					const errorText = await response.text();
					// Node.js filesystem operations are synchronous with ...Sync suffix
					// (unlike Python where synchronous is default)
					throw new Error(`Error getting price for ${resolvedSymbol}: ${response.status} ${errorText}`);
				}
				// TypeScript requires type assertion (as any) here because the JSON structure
				// isn't known at compile time (Python doesn't need this)
				const data = await response.json() as any;
				const price = data.price;
				// MCP response format is more explicit in TypeScript due to static typing
				return { content: [{ type: "text", text: `The current price of ${resolvedSymbol} is ${price}` }] };
			}
		);
	}

	
}

export default {
	fetch(request: Request, env: Env, ctx: ExecutionContext) {
		const url = new URL(request.url);

		if (url.pathname === "/sse" || url.pathname === "/sse/message") {
			// @ts-ignore
			return MyMCP.serveSSE("/sse").fetch(request, env, ctx);
		}

		if (url.pathname === "/mcp") {
			// @ts-ignore
			return MyMCP.serve("/mcp").fetch(request, env, ctx);
		}

		return new Response("Not found", { status: 404 });
	},
};
