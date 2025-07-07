# Houdini_MCP
## HoudiniMCP – Connect Houdini to Claude via Model Context Protocol

---
HoudiniMCP allows you to control SideFX Houdini from Claude using the Model Context Protocol (MCP). It consists of:

A Houdini plugin (Python package) that listens on a local port (default localhost:9876) and handles commands (creating and modifying nodes, executing code, etc.).
An MCP bridge script you run using your virtual environment (e.g., python houdini_mcp_server.py) that communicates via stdin/stdout with Claude and via TCP with Houdini.

---

## Requirements
- SideFX Houdini
- Claude Desktop(latest version)

---

## 1. Houdini MCP Plugin Installation

### 1.1 Folder Layout
Create a folder in your Houdini scripts directory:
C:/Users/<YourUserName>/Documents/houdini##.#/scripts/python/houdinimcp/

Inside `houdinimcp/`, place:
- `__init__.py` - handles plugin initialization (start/stop server)
- `server.py` - defines the `HoudiniMCPServer` (listening on default port `9876`)
- `houdini_mcp_server.py` – optional bridging script (some prefer a separate location)
- `pyproject.toml`


### 1.2 Set Up Virtual Environment(CMD or PowerShell)
In the `houdinimcp/` directory, set up a Python virtual environment and install the dependencies:
~~~ terminal
cd C:/Users/<YourUserName>/Documents/houdini##.#/scripts/python/houdinimcp
python -m venv .venv
.\.venv\Scripts\activate
pip install fastapi requests openai
~~~

### 1.3 Shelf Tool
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

### 1.4 Packages Integration
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


## 2. Telling Claude for Desktop to Use Your Script

Go to File > Settings > Developer > Edit Config > Open or create: claude_desktop_config.json

Add an entry:
~~~ python
{
  "mcpServers": {
    "houdini": {
      "command": "C:/Users/<YourUserName>/Documents/houdini##.#/scripts/python/houdinimcp/.venv/Scripts/python.exe",
      "args": [
        "C:/Users/<YourUserName>/Documents/houdini##.#/scripts/python/houdinimcp/houdini_mcp_server.py"
      ]
    }
  }
}
~~~
If the server starts but Claude fails to load the MCP connection, make sure Claude is using the same Python version as your virtual environment. Use:
~~~ python
  python -c "import sys; print(sys.executable)"
~~~
to find python, and replace "python" with the path you got.

## 3. Use Cursor
Go to Settings > MCP > add new MCP server add the same entry in claude_desktop_config.json you might need to stop claude and restart houdini and the server

## If the installation is successful, Claude should appear like this:
<img width="489" alt="Screenshot 2025-07-07 at 2 34 42 PM" src="https://github.com/user-attachments/assets/249f24b0-2757-43a6-af53-80874b20d6e6" />

