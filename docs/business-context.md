# Foundation Assessment Business Context

This page explains the simulated PostgreSQL fields used by the local Foundation/Risk workflow.
The screen at `/mock/foundation` lets an analyst edit these fields before running the chain so
small business-context changes can be tested.

## Vendor Profile

- **Vendor name**: The business name shown in the report.
- **Vendor type**: The kind of supplier, such as SaaS provider, hosting provider, consultancy, or HR provider. This helps decide which questionnaire answers and risks are relevant.
- **Business relationship**: What the vendor does for the organization. This is important because the same technical weakness can have different business impact depending on the relationship.
- **Country / region**: Where the vendor primarily operates. This can influence regulatory exposure, data-transfer concerns, and operational resilience expectations.
- **Services in scope**: The services being assessed. Risk evaluation should stay tied to these services, not the whole vendor company unless explicitly stated.

## Tier Context

The tier represents the risk the vendor relationship creates for the organization. In the prototype
the level is supplied by the simulated database, but in production it is expected to be calculated by
application logic and then stored with the assessment.

- **Tier level**: A level from 1 to 4. Lower numbers represent higher criticality in this prototype.
- **Tier definition**: Human-readable explanation of why the vendor has this tier.
- **Company size**: Indicates operational footprint and support capacity.
- **Sensitive data access**: Whether the vendor processes sensitive business, customer, employee, or regulated data.
- **Privileged access**: Whether the vendor can administer or deeply access internal systems.
- **Geolocation**: The region where the vendor operates or stores/processes data.

## Questionnaire Result Context

Each questionnaire result is linked to one security control. The answer is assessed by the analyst
and then used by the workflow to decide which gaps need risk evaluation.

- **Question text**: What the vendor was asked.
- **Vendor answer**: The vendor's direct response.
- **Vendor comment**: Free text supplied by the vendor. This is sanitized before model use.
- **Analyst comment**: Analyst interpretation or challenge. This is sanitized before model use.
- **Compliance**: Whether the response is full, partial, no, not applicable, or unknown compliance.
- **Maturity**: How mature the implemented control appears to be.
- **Evidence descriptions**: Analyst-visible descriptions of uploaded evidence. In production,
  uploads are sandboxed/scanned and only PDF/JPG/PNG evidence is accepted.

## How These Fields Affect The Workflow

1. The source adapter reads the simulated database row and creates a canonical assessment packet.
2. The workflow cleans comments and classifies answers.
3. Full compliance is treated as a strength.
4. Partial or no compliance becomes a weakness.
5. Each weakness is turned into risk questions.
6. Standards evidence is retrieved from Qdrant, BM25, and Neo4j.
7. The selected model writes risk/control answers only from the supplied assessment data and retrieved evidence.
8. The final result is packaged for later PostgreSQL storage after analyst review.
