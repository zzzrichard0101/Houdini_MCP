# Houdini_MCP
## HoudiniMCP – Connect Houdini to Claude via Model Context Protocol

---
HoudiniMCP allows you to control SideFX Houdini from Claude using the Model Context Protocol (MCP). It consists of:

A Houdini plugin (Python package) that listens on a local port (default localhost:9876) and handles commands (creating and modifying nodes, executing code, etc.).
An MCP bridge script you run via uv (or system Python) that communicates via stdin/stdout with Claude and TCP with Houdini.
Below are the complete instructions for setting up Houdini, uv, and Claude Desktop.

---

## Requierments
- SideFX Houdini
- Claude Desktop(latest version)

---

## Houdini MCP Pligin Installation
