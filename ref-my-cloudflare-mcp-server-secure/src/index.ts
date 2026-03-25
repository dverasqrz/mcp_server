import OAuthProvider from "@cloudflare/workers-oauth-provider";
import { McpAgent } from "agents/mcp";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { Octokit } from "octokit";
import { GitHubHandler } from "./github-handler";


function getSymbolFromName(name: string): string {
  if (["bitcoin", "btc"].includes(name.toLowerCase())) return "BTCUSDT";
  if (["ethereum", "eth"].includes(name.toLowerCase())) return "ETHUSDT";
  // TypeScript's ternary operator - like Python's `x if condition else y`
  // The explicit type check is needed because TypeScript is more strict about types
  return typeof name === "string" ? name.toUpperCase() : String(name).toUpperCase();
}

// Context from the auth process, encrypted & stored in the auth token
// and provided to the DurableMCP as this.props
type Props = {
  login: string;
  name: string;
  email: string;
  accessToken: string;
};

const ALLOWED_USERNAMES = new Set<string>([
  // Add GitHub usernames of users who should have access to the image generation tool
  // For example: 'yourusername', 'coworkerusername'
]);

export class MyMCP extends McpAgent<Env, {}, Props> {
  server = new McpServer({
    name: "Binance Oauth MCP Server",
    version: "1.0.0",
  });

  async init() {
    // Hello, world!
    this.server.tool("add", "Add two numbers the way only MCP can", { a: z.number(), b: z.number() }, async ({ a, b }) => ({
      content: [{ type: "text", text: String(a + b) }],
    }));

    // Use the upstream access token to facilitate tools
    this.server.tool("userInfoOctokit", "Get user info from GitHub, via Octokit", {}, async () => {
      const octokit = new Octokit({ auth: this.props.accessToken });
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(await octokit.rest.users.getAuthenticated()),
          },
        ],
      };
    });

    // Dynamically add tools based on the user's login. In this case, I want to limit
    // access to my Image Generation tool to just me
    if (ALLOWED_USERNAMES.has(this.props.login)) {
      this.server.tool(
        "generateImage",
        "Generate an image using the `flux-1-schnell` model. Works best with 8 steps.",
        {
          prompt: z.string().describe("A text description of the image you want to generate."),
          steps: z
            .number()
            .min(4)
            .max(8)
            .default(4)
            .describe(
              "The number of diffusion steps; higher values can improve quality but take longer. Must be between 4 and 8, inclusive.",
            ),
        },
        async ({ prompt, steps }) => {
          const response = await this.env.AI.run("@cf/black-forest-labs/flux-1-schnell", {
            prompt,
            steps,
          });

          return {
            content: [{ type: "image", data: response.image!, mimeType: "image/jpeg" }],
          };
        },
      );
    }

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


export default new OAuthProvider({
  apiRoute: "/sse",
  apiHandler: MyMCP.mount("/sse") as any,
  defaultHandler: GitHubHandler as any,
  authorizeEndpoint: "/authorize",
  tokenEndpoint: "/token",
  clientRegistrationEndpoint: "/register",
});
