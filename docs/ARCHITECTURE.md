# Architecture and System Design (C4 Style)

This document describes the system using C4-inspired views and a runtime flow diagram.
All diagrams use color/styling so responsibilities and trust boundaries are easy to scan.

## C4 Level 1 - System Context

```mermaid
flowchart LR
    user["User<br/>(Uploads DOCX, downloads ZIP)"]
    maintainer["Maintainer / Developer<br/>(Operates services)"]

    system["Word-to-LaTeX Platform<br/>DOCX -> LaTeX conversion service"]

    office["LibreOffice / soffice<br/>(Reference PDF generation)"]
    gemini["Gemini API<br/>(Optional enhancement)"]
    localLlm["Local GGUF Model<br/>(Optional routing/classification)"]

    user -->|Upload DOCX, get ZIP/JSON| system
    maintainer -->|Configure + deploy| system

    system -->|Convert DOCX to PDF (optional)| office
    system -->|LLM prompt/response (optional)| gemini
    system -->|Inference (optional)| localLlm

    classDef actor fill:#FFE6C9,stroke:#D97706,color:#7C2D12,stroke-width:2px;
    classDef system fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A,stroke-width:2.5px;
    classDef external fill:#E0F2FE,stroke:#0891B2,color:#164E63,stroke-width:2px;

    class user,maintainer actor;
    class system system;
    class office,gemini,localLlm external;
```

## C4 Level 2 - Container Diagram

```mermaid
flowchart TB
    subgraph Client["Client Layer"]
      direction LR
      nextUi["Next.js UI"]
      browserUi["Built-in Web UI Client<br/>Browser"]
    end

    subgraph Api["FastAPI Application"]
      direction LR
      app["FastAPI App<br/>server.main"]
      convApi["Conversion API Router<br/>/v1/convert and /v1/convert/json"]
      webApi["Web Router<br/>/ and /stats"]
      svc["Conversion Service<br/>convert_document()"]
      state["In-memory State<br/>RateLimiter and UsageStats"]
    end

    subgraph Processing["Conversion Processing"]
      direction LR
      core["wordtolatex Core Library<br/>Parser, Layout, Calibration,<br/>Policy, Generation"]
      temp["Temp Workspace<br/>input.docx, ref.pdf,<br/>output, zip"]
    end

    subgraph External["Optional Integrations"]
      direction LR
      soffice["LibreOffice or soffice"]
      local["Local GGUF Model"]
      gemini["Gemini API"]
    end

    nextUi --> app
    browserUi --> app

    app --> convApi
    app --> webApi

    convApi --> svc
    convApi --> state
    webApi --> state

    svc --> core
    core --> temp

    svc -."ref PDF generation".-> soffice
    svc -."LLM routing".-> local
    svc -."AI enhancement".-> gemini

    classDef client fill:#FEF3C7,stroke:#D97706,color:#78350F,stroke-width:2px;
    classDef api fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A,stroke-width:2px;
    classDef processing fill:#DCFCE7,stroke:#16A34A,color:#14532D,stroke-width:2px;
    classDef ext fill:#E0F2FE,stroke:#0891B2,color:#164E63,stroke-width:2px;

    class nextUi,browserUi client;
    class app,convApi,webApi,svc,state api;
    class core,temp processing;
    class soffice,local,gemini ext;
```

## C4 Level 3 - Component Diagram (Conversion Path)

```mermaid
flowchart LR
    request["POST /v1/convert<br/>(docx, ref_pdf?, options_json?)"]

    options["parse_options()<br/>(guarded by ALLOW_USER_OPTIONS)"]
    limiter["enforce_rate_limit()"]

    convertSvc["convert_document()<br/>(orchestration)"]

    parseDocx["parse_docx()"]
    refPdf["generate_reference_pdf()<br/>(if ref_pdf missing)"]
    analyze["analyze_document()<br/>(layout metrics)"]
    applyProfile["apply_profile()<br/>(auto/manual/calibrated)"]
    headerFallback["apply_header_image_fallback()"]
    assignPolicy["decide_policy() per block"]
    render["dynamic_generate() or generate_latex()"]
    package["zip_directory(output/)<br/>-> wordtolatex_output.zip"]

    response["FileResponse (ZIP)<br/>or ConversionResponse (JSON)"]

    request --> options --> limiter --> convertSvc
    convertSvc --> parseDocx
    parseDocx --> refPdf
    refPdf --> analyze --> applyProfile --> headerFallback --> assignPolicy --> render --> package --> response

    classDef ingress fill:#FFEDD5,stroke:#EA580C,color:#7C2D12,stroke-width:2px;
    classDef guard fill:#FDE68A,stroke:#CA8A04,color:#713F12,stroke-width:2px;
    classDef orchestrator fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A,stroke-width:2.5px;
    classDef pipeline fill:#DCFCE7,stroke:#16A34A,color:#14532D,stroke-width:2px;
    classDef egress fill:#E0E7FF,stroke:#4F46E5,color:#312E81,stroke-width:2px;

    class request ingress;
    class options,limiter guard;
    class convertSvc orchestrator;
    class parseDocx,refPdf,analyze,applyProfile,headerFallback,assignPolicy,render,package pipeline;
    class response egress;
```

## Runtime Request Lifecycle (System Design View)

```mermaid
sequenceDiagram
    autonumber
    participant U as User / UI
    participant A as FastAPI Router
    participant S as Conversion Service
    participant C as Core Engine
    participant X as External Tools

    rect rgb(255, 244, 229)
      U->>A: POST /v1/convert (docx, options, ref_pdf?)
      A->>A: Validate upload + rate limit
    end

    rect rgb(232, 244, 255)
      A->>S: convert_document(...)
      S->>C: parse_docx(input.docx)
      alt ref_pdf missing and AUTO_REFERENCE_PDF=true
        S->>X: Run soffice to generate ref.pdf
      end
      S->>C: analyze_document(ref.pdf)
      S->>C: apply_profile + header_fallback + policy
      S->>C: generate_latex(dynamic or template)
      S->>S: write output + zip_directory
    end

    rect rgb(235, 255, 240)
      S-->>A: zip_path + metadata
      A-->>U: 200 OK (application/zip)
    end
```

## Notes

- The current implementation keeps rate limiting and stats in process memory.
- `options_json` is intentionally ignored unless `ALLOW_USER_OPTIONS=true`.
- External integrations are optional and controlled by settings.
