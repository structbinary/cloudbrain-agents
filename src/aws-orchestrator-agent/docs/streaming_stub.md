# WebSocket Streaming Stub Implementation

## Overview
This document describes the stub implementation for real-time streaming of A2A task updates via WebSocket in the AWS Orchestrator Agent project. The stub is intended as a placeholder for future integration with the actual event queue or task update mechanism.

## Purpose
- Demonstrate how to stream real-time updates to clients using a WebSocket endpoint.
- Provide a clear integration point for connecting to the real A2A event or task update system.
- Allow developers to test the streaming infrastructure end-to-end before full backend integration.

## Where to Find the Code
- **WebSocket Endpoint:**
  - File: `aws_orchestrator_agent/core/agent_card_server.py`
  - Endpoint: `/stream/{context_id}`
  - Handler: `stream_updates`
- **Stub Streaming Generator:**
  - File: `aws_orchestrator_agent/core/a2a_integration.py`
  - Class: `A2AIntegrationManager`
  - Method: `async def stream_task_updates(self, context_id: str)`

## How the Stub Works
- When a client connects to `/stream/{context_id}` via WebSocket, the server:
  1. Instantiates a new `A2AIntegrationManager` (in production, use a shared, initialized instance).
  2. Calls `stream_task_updates(context_id)`, which yields 10 fake updates (one per second) as JSON messages.
  3. Sends each update to the WebSocket client.
  4. Closes the connection after all updates are sent.

- **Example update message:**
  ```json
  {
    "context_id": "abc123",
    "timestamp": 1721539200.123,
    "message": "Stub streaming update 1"
  }
  ```

## How to Replace the Stub
- Implement real event streaming in `A2AIntegrationManager.stream_task_updates`:
  - Connect to your event queue, task lifecycle manager, or other backend system.
  - Yield updates as they arrive for the given `context_id`.
  - Ensure updates are sent as JSON-serializable Python dicts.
- Update the WebSocket handler to use your real, shared integration manager instance (not a new one per connection).

## Usage Instructions for Developers
1. **Start the server** (see your project README for details).
2. **Connect a WebSocket client** to `ws://<host>:<port>/stream/<context_id>`.
3. **Observe streaming updates**: You should receive 10 stub messages, one per second.
4. **Replace the stub** with real event streaming when ready.

## TODOs
- Integrate with the actual A2A event queue or task update mechanism.
- Use a shared, initialized `A2AIntegrationManager` instance in production.
- Add authentication and error handling as needed for production use. 