# AWS Provider Documentation Search Tool - Workflow Diagram

## Overview
This tool searches the Terraform AWS provider documentation by fetching content directly from the official GitHub repository and parsing markdown files to extract structured information about AWS resources and data sources.

## Architecture & Workflow

```mermaid
graph TD
    A[User Request] --> B[search_aws_provider_docs MCP Tool]
    B --> C[search_aws_provider_docs_impl]
    
    C --> D{Input Validation}
    D -->|Invalid| E[Return Error Result]
    D -->|Valid| F[Generate Correlation ID]
    
    F --> G{Asset Type Check}
    G -->|'both'| H[Search Both Resource & Data Source]
    G -->|'resource'| I[Search Resource Only]
    G -->|'data_source'| J[Search Data Source Only]
    
    H --> K[fetch_github_documentation - Resource]
    H --> L[fetch_github_documentation - Data Source]
    I --> K
    J --> L
    
    K --> M{Check Cache}
    L --> M
    M -->|Cache Hit| N[Return Cached Result]
    M -->|Cache Miss| O[resource_to_github_path]
    
    O --> P[Construct GitHub URL]
    P --> Q[Fetch Markdown from GitHub]
    Q --> R{HTTP Response}
    R -->|200 OK| S[parse_markdown_documentation]
    R -->|Error| T[Return None]
    
    S --> U[Extract Title & Description]
    S --> V[Extract Example Usage]
    S --> W[Extract Arguments Reference]
    S --> X[Extract Attributes Reference]
    
    U --> Y[Create TerraformAWSProviderDocsResult]
    V --> Y
    W --> Y
    X --> Y
    
    Y --> Z[Cache Result]
    Z --> AA[Return Structured Result]
    
    N --> BB[Format Results]
    AA --> BB
    T --> CC[Return Not Found Result]
    
    BB --> DD[Final Response to User]
    CC --> DD
    E --> DD
```

## Detailed Logic Flow

### 1. **Entry Point & Validation**
```python
@mcp.tool(name='SearchAwsProviderDocs')
async def search_aws_provider_docs(
    asset_name: str,
    asset_type: str = 'resource'
) -> List[TerraformAWSProviderDocsResult]:
    return await search_aws_provider_docs_impl(asset_name, asset_type)
```

### 2. **Core Implementation Logic**
```python
async def search_aws_provider_docs_impl(
    asset_name: str, 
    asset_type: str = 'resource', 
    cache_enabled: bool = False
):
    # 1. Input validation
    # 2. Generate correlation ID for logging
    # 3. Determine search strategy based on asset_type
    # 4. Fetch documentation from GitHub
    # 5. Parse and structure results
    # 6. Return formatted results
```

### 3. **GitHub Path Construction**
```python
def resource_to_github_path(asset_name: str, asset_type: str):
    # 1. Sanitize input (prevent path traversal)
    # 2. Remove 'aws_' prefix if present
    # 3. Determine document type ('r' for resource, 'd' for data source)
    # 4. Construct file path: {doc_type}/{resource_name}.html.markdown
    # 5. Build full GitHub raw URL
```

### 4. **Documentation Fetching**
```python
def fetch_github_documentation(asset_name: str, asset_type: str, cache_enabled: bool):
    # 1. Check in-memory cache first
    # 2. Construct GitHub URL using resource_to_github_path
    # 3. Make HTTP request to GitHub raw content
    # 4. Handle timeouts and errors
    # 5. Cache successful results
    # 6. Parse markdown content
```

### 5. **Markdown Parsing**
```python
def parse_markdown_documentation(content: str, asset_name: str, url: str):
    # 1. Extract title from first heading
    # 2. Extract description from resource section
    # 3. Parse Example Usage section for code snippets
    # 4. Parse Argument Reference section for parameters
    # 5. Parse Attribute Reference section for outputs
    # 6. Return structured dictionary
```

## Key Design Patterns

### 1. **Caching Strategy**
- **In-memory cache**: `_GITHUB_DOC_CACHE` dictionary
- **Cache key**: `{asset_name}_{asset_type}`
- **Cache invalidation**: Manual (no automatic expiration)

### 2. **Error Handling**
- **Input validation**: Sanitize asset names, validate asset types
- **Network errors**: Handle timeouts and HTTP errors gracefully
- **Parsing errors**: Return partial results with error descriptions
- **Security**: Prevent path traversal attacks

### 3. **Logging & Observability**
- **Correlation IDs**: Track requests across function calls
- **Structured logging**: Use loguru with detailed formatting
- **Performance metrics**: Track execution times
- **Error tracking**: Log errors without exposing sensitive data

### 4. **Flexible Search Strategy**
- **Asset type handling**: Support 'resource', 'data_source', or 'both'
- **Prefix handling**: Automatically handle 'aws_' prefix
- **Fallback mechanisms**: Try multiple paths for comprehensive results

## Data Flow

### Input Processing
1. **Asset Name**: `aws_s3_bucket` → `s3_bucket` (prefix removal)
2. **Asset Type**: Determines GitHub path construction
3. **Validation**: Ensures safe input parameters

### GitHub Integration
1. **URL Construction**: `https://raw.githubusercontent.com/hashicorp/terraform-provider-aws/main/website/docs/r/s3_bucket.html.markdown`
2. **Content Fetching**: HTTP GET request with timeout
3. **Response Handling**: Status code validation

### Content Parsing
1. **Title Extraction**: First `#` heading
2. **Description**: Content after resource title
3. **Examples**: Code blocks in "Example Usage" section
4. **Arguments**: Bullet points in "Argument Reference" section
5. **Attributes**: Bullet points in "Attribute Reference" section

### Output Structuring
```python
TerraformAWSProviderDocsResult(
    asset_name="aws_s3_bucket",
    asset_type="resource",
    description="Provides a S3 bucket resource...",
    url="https://raw.githubusercontent.com/...",
    example_usage=[{"title": "Basic Usage", "code": "resource \"aws_s3_bucket\"..."}],
    arguments=[{"name": "bucket", "description": "Name of the bucket"}],
    attributes=[{"name": "id", "description": "Name of the bucket"}]
)
```

## Performance Optimizations

1. **Caching**: In-memory cache for repeated queries
2. **Parallel Processing**: When searching 'both' types
3. **Timeout Management**: 10-second timeout for GitHub requests
4. **Content Length Limits**: Preview logging for large responses

## Security Considerations

1. **Input Sanitization**: Regex validation for asset names
2. **Path Traversal Prevention**: URL construction validation
3. **Error Information**: Limited error details in responses
4. **Domain Validation**: Ensure URLs point to expected GitHub domain

This architecture provides a robust, scalable solution for searching AWS provider documentation with comprehensive error handling, caching, and structured output. 