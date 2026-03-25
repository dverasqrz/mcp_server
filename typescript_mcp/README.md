# TypeScript MCP Project

This project implements a Model Context Protocol server using TypeScript.

## Project Configuration

The project uses TypeScript with a modern configuration optimized for Node.js development. Here's a detailed explanation of our TypeScript configuration:

### TypeScript Configuration Explained

Our `tsconfig.json` sets up a development environment that:
1. Uses modern JavaScript features
2. Implements strong type checking
3. Has clear module resolution rules
4. Maintains cross-platform compatibility

#### Key Configuration Elements:

- **Target (ES2022)**: Similar to specifying Python version, this determines which JavaScript features we can use. ES2022 gives us access to modern language features.

- **Module System**: Uses Node16's module system, which is similar to Python's import system. This determines how code modules work together and how imports are resolved.

- **Directory Structure**:
  - `src/`: Source code directory (TypeScript files)
  - `dist/`: Compiled output directory (JavaScript files)
  - `node_modules/`: External dependencies (like Python's site-packages)

- **Type Checking**: Uses strict mode, similar to running Python with strict type checking (`mypy --strict`). This catches more errors during development rather than at runtime.

- **Module Resolution**: Uses Node16's algorithm for finding modules, similar to how Python uses `sys.path` to resolve imports.

### Package Configuration (package.json) Explained

The `package.json` file is similar to Python's `setup.py` and `pyproject.toml` combined. Here's what each section means for Python developers:

#### Core Package Information
- `name`: Package name in npm's registry (like Python package name in PyPI)
  - The `@username/` prefix is for scoped packages (similar to Python's namespace packages)
- `version`: Semantic version (same as in `setup.py`)
  - This version is imported in `binance_mcp.ts` to keep versions in sync
- `description`: Package description (like in `setup.py`)
- `main`: The main entry point when this package is imported (similar to `__init__.py`)
- `bin`: Makes the package executable from command line (like Python's `console_scripts`)

#### Dependencies
- `dependencies`: Runtime dependencies (like `requirements.txt` or `install_requires` in `setup.py`)
  - `@modelcontextprotocol/sdk`: The MCP SDK package
  - `zod`: Runtime type checking (similar to Python's `typing` module)

- `devDependencies`: Development-only dependencies (like `dev-requirements.txt`)
  - `@types/node`: TypeScript type definitions for Node.js
  - `typescript`: The TypeScript compiler

#### Scripts
- `scripts`: Automation commands (like Makefile targets or `setup.py` commands)
  - `build`: Compiles TypeScript to JavaScript
  - `prepare`: Automatically runs build before package installation

#### Environment
- `engines`: Node.js version requirements (like `python_requires` in `setup.py`)

## Getting Started

1. Install dependencies:
```bash
npm install
```

2. Build the project:
```bash
npm run build
```

3. Run the server:
```bash
npx typescript_mcp
```

## Dependencies

- `@modelcontextprotocol/sdk`: Core MCP functionality
- `zod`: Runtime type checking
- `typescript`: Development dependency for TypeScript compilation
- `@types/node`: Type definitions for Node.js

## Development Notes

The strict configuration ensures type safety throughout the project. When adding new code, make sure to:
1. Use proper type annotations
2. Handle all potential error cases
3. Follow the module import/export patterns established in existing files


```