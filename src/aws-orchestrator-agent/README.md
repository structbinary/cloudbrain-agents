# AWS Orchestrator Agent

## Multi-Agent LangGraph Workflow

This project implements a modular, multi-agent orchestration system for Terraform module generation and integration, using LangGraph and the A2A protocol.

### Workflow Overview

```mermaid
flowchart TD
    U["User"]
    S["aws_orchestrator_supervisor_agent\n(Clarifies, routes, has knowledge base)"]
    G["Module Generation Agent"]
    I["Integration Agent"]
    V["Validation Agent"]

    U --> S
    S -- "New module request" --> G
    S -- "Existing module change" --> I
    G --> V
    I --> V
    V --> S
    S --> U

    %% Optional: Human-in-the-Loop for clarification
    subgraph Human["Human-in-the-Loop (Clarification)"]
        H["User Clarification Loop"]
    end
    S -.->|"If task unclear"| H
    H -.->|"Clarified details"| S
```

### Flow Explanation
- **User** contacts the `aws_orchestrator_supervisor_agent`.
- **Supervisor agent** clarifies requirements with the user if the task is unclear (human-in-the-loop), has access to the knowledge base, and routes the request:
  - To **Module Generation Agent** for new modules.
  - To **Integration Agent** for existing module changes.
- After either generation or integration, the **Validation Agent** is always invoked.
- The **Validation Agent** returns results to the supervisor, which then responds to the user.
