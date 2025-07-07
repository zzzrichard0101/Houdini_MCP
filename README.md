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

## Houdini MCP Plugin Installation

### 1. Folder Layout
Creat a folder in your Houdini scripts directory:
C:/Users/YourUserName/Documents/houdini##.#/scripts/python/houdinimcp/

Inside **houdinimcp/**, place:
- __init__.py - handles plugin initialization (start/stop server)
- server.py - defines the HoudiniMCPServer (listening on default port 9876)
- houdini_mcp_server.py – optional bridging script (some prefer a separate location)
- pyproject.toml

(If you prefer, houdini_mcp_server.py can live elsewhere. As long as you know its path for running with uv.)

### 2. Shelf Tool
create a Shelf Tool to toggle the server in Houdini:
1. Right-click a shelf → "New Shelf..."
Name it "MCP" or something similar
2. Right-click again → "New Tool..." Name: "Toggle MCP Server" Label: "MCP"
3. Under Script, insert something like:
~~~ python
   import hou
   import houdinimcp

   if hasattr(hou.session, "houdinimcp_server") and hou.session.houdinimcp_server:
       houdinimcp.stop_server()
       hou.ui.displayMessage("Houdini MCP Server stopped")
   else:
       houdinimcp.start_server()
       hou.ui.displayMessage("Houdini MCP Server started on localhost:9876")
~~~

### 3. Packages Integration
If you want Houdini to auto-load your plugin at startup, create a package file named houdinimcp.json in the Houdini packages folder (e.g. C:/Users/YourUserName/Documents/houdini19.5/packages/):
~~~ python
{
  "path": "$HOME/houdini19.5/scripts/python/houdinimcp",
  "load_package_once": true,
  "version": "0.1",
  "env": [
    {
      "PYTHONPATH": "$PYTHONPATH;$HOME/houdini19.5/scripts/python"
    }
  ]
}
~~~




