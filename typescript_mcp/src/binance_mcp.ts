#!/usr/bin/env node

// In TypeScript, we use ES Module imports (similar to Python's `from x import y`)
// The `.js` extension is required here even though we're in a `.ts` file - this is a TypeScript quirk
import { McpServer, ResourceTemplate } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
// zod is TypeScript's runtime type checking library - similar to Python's typing + runtime checks
import { z } from "zod";
// Node.js built-in modules (like Python's standard library)
import fs from "fs";
import path from "path";

// Import version from package.json
// We need to use require here because JSON imports need special TypeScript config
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { version } = require("../package.json");

// __dirname is Node.js's way to get current directory (similar to Python's __file__)
// Unlike Python's os.path.join, Node.js path.join handles cross-platform paths automatically
const ACTIVITY_LOG_FILE = path.join(__dirname, "../../activity.log");

// TypeScript function with type annotation (similar to Python's type hints)
// The `: string` after the parameter and before the block is the return type annotation
function getSymbolFromName(name: string): string {
    if (["bitcoin", "btc"].includes(name.toLowerCase())) return "BTCUSDT";
    if (["ethereum", "eth"].includes(name.toLowerCase())) return "ETHUSDT";
    // TypeScript's ternary operator - like Python's `x if condition else y`
    // The explicit type check is needed because TypeScript is more strict about types
    return typeof name === "string" ? name.toUpperCase() : String(name).toUpperCase();
}

// Create MCP server instance - similar to FastMCP in Python
// TypeScript uses object literal syntax for configuration (like Python's kwargs)
const server = new McpServer({
    name: "Binance MCP",
    version  // Using version from package.json (shorthand object property)
});

// Tool registration in TypeScript - equivalent to Python's @mcp.tool() decorator
// Instead of function decorators, we use a method call pattern common in TypeScript
server.tool(
    // Tool name
    "get_price",
    // Parameter schema using zod (similar to Python's type hints but with runtime validation)
    { symbol: z.string() },
    // Async handler function (similar to Python's async def)
    // Arrow function syntax is common in TypeScript: ({params}) => { body }
    async ({ symbol }) => {
        const resolvedSymbol = getSymbolFromName(symbol);
        const url = `https://api.binance.com/api/v3/ticker/price?symbol=${resolvedSymbol}`;
        // fetch is built into modern Node.js (similar to Python's requests.get)
        const response = await fetch(url);
        if (!response.ok) {
            const errorText = await response.text();
            // Node.js filesystem operations are synchronous with ...Sync suffix
            // (unlike Python where synchronous is default)
            fs.appendFileSync(
                ACTIVITY_LOG_FILE,
                `Error getting price for ${resolvedSymbol}: ${response.status} ${errorText}\n`
            );
            throw new Error(`Error getting price for ${resolvedSymbol}: ${response.status} ${errorText}`);
        }
        // TypeScript requires type assertion (as any) here because the JSON structure
        // isn't known at compile time (Python doesn't need this)
        const data = await response.json() as any;
        const price = data.price;
        fs.appendFileSync(
            ACTIVITY_LOG_FILE,
            `Successfully got price for ${resolvedSymbol}. Current price is ${price}. Current time is ${new Date().toISOString()}\n`
        );
        // MCP response format is more explicit in TypeScript due to static typing
        return { content: [{ type: "text", text: `The current price of ${resolvedSymbol} is ${price}` }] };
    }
);

// Similar pattern for price change endpoint
server.tool(
    "get_price_price_change",
    { symbol: z.string() },
    async ({ symbol }) => {
        const resolvedSymbol = getSymbolFromName(symbol);
        const url = `https://data-api.binance.vision/api/v3/ticker/24hr?symbol=${resolvedSymbol}`;
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Error getting price change for ${resolvedSymbol}: ${response.status}`);
        }
        const data = await response.json() as any;
        return { content: [{ type: "text", text: JSON.stringify(data) }] };
    }
);

// Resource registration - similar to FastMCP's resource handling
// TypeScript requires more explicit template and type definitions
server.resource(
    "file://activity.log",
    // ResourceTemplate is a TypeScript class that defines the resource interface
    new ResourceTemplate("file://activity.log", { list: undefined }),
    async () => {
        // Node.js pattern: check existence before reading
        if (!fs.existsSync(ACTIVITY_LOG_FILE)) return { contents: [{ uri: "file://activity.log", text: "" }] };
        const text = fs.readFileSync(ACTIVITY_LOG_FILE, "utf-8");
        return { contents: [{ uri: "file://activity.log", text }] };
    }
);

// Dynamic resource with URL parameters
// Similar to FastMCP's dynamic resources but with explicit type handling
server.resource(
    "resource://crypto_price/{symbol}",
    new ResourceTemplate("resource://crypto_price/{symbol}", { list: undefined }),
    // Resource handlers receive URI and variables (like FastMCP's path parameters)
    async (uri, variables) => {
        // Type assertion needed because TypeScript can't infer the type from the template
        const symbol = variables.symbol as string;
        const resolvedSymbol = getSymbolFromName(symbol);
        const url = `https://api.binance.com/api/v3/ticker/price?symbol=${resolvedSymbol}`;
        const response = await fetch(url);
        if (!response.ok) {
            return { contents: [{ uri: uri.href, text: `Error getting price for ${resolvedSymbol}` }] };
        }
        const data = await response.json() as any;
        return { contents: [{ uri: uri.href, text: `The current price of ${resolvedSymbol} is ${data.price}` }] };
    }
);


// Main server startup function
// TypeScript/Node.js typically uses async/await for startup (similar to Python's async main)
async function startServer() {
    // Ensure log file exists (Node.js pattern - Python would typically use 'a' mode)
    if (!fs.existsSync(ACTIVITY_LOG_FILE)) {
        fs.writeFileSync(ACTIVITY_LOG_FILE, "");
    }
    // Create transport instance (equivalent to transport="stdio" in Python)
    const transport = new StdioServerTransport();
    await server.connect(transport);
    console.error(`Binance MCP Server v${version} started`);
    console.error("Tools: get_price, get_price_price_change");
    console.error("Resources: file://activity.log, resource://crypto_price/{symbol}");
}

// Error handling pattern in Node.js
// Similar to Python's if __name__ == "__main__" but with promise error handling
startServer().catch(err => {
    console.error("Error starting server:", err);
    process.exit(1);
});
