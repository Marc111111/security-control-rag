# Standards Source Manifest

Downloaded locally on 2026-06-13 for ingestion into `storage/chroma`.

Large standards files in this folder are intentionally not committed to Git. Re-download from the
source URLs below when rebuilding the local corpus.

## Downloaded Files

- `nist-sp-800-53-rev5.pdf`
  - Source: `https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf`
  - Notes: NIST SP 800-53 Revision 5 PDF.
- `nist-sp-800-53-rev5-control-catalog.xlsx`
  - Source: `https://csrc.nist.gov/CSRC/media/Publications/sp/800-53/rev-5/final/documents/sp800-53r5-control-catalog.xlsx`
  - Notes: Structured control catalog for NIST SP 800-53 Revision 5.
- `nist-csf-2.0.pdf`
  - Source: `https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf`
  - Notes: NIST Cybersecurity Framework 2.0, CSWP 29, February 26, 2024.
- `scf-2026.1.1.xlsx`
  - Source: `https://raw.githubusercontent.com/securecontrolsframework/securecontrolsframework/main/Secure%20Controls%20Framework%20%28SCF%29%20-%202026.1.1.xlsx`
  - Notes: Secure Controls Framework spreadsheet, 2026.1.1.
- `scf-overview-practitioner-guidebook-2026.2.pdf`
  - Source: `https://raw.githubusercontent.com/securecontrolsframework/securecontrolsframework/main/SCF%20Overview%20%26%20Practitioner%20Guidebook%20%282026.2%29.pdf`
  - Notes: Secure Controls Framework overview and practitioner guidebook, 2026.2.
- `cis-controls-v8.1-official-page.html`
  - Source: `https://www.cisecurity.org/controls/v8-1`
  - Notes: Official CIS Controls v8.1 landing page.
- `cis-controls-v8.1-official-page.txt`
  - Source: text extraction from `cis-controls-v8.1-official-page.html`
  - Notes: Plain text form so the current ingestion pipeline can index official CIS page content.
- `cis-controls-v8.1-guide-mirror.pdf`
  - Source: `https://etir.unb.br/wp-content/uploads/2024/10/CIS_Controls__v8.1_Guide__2024_06.pdf`
  - Notes: Public mirror of the CIS Controls v8.1 Guide. CIS's official direct PDF endpoint returned
    a form/error payload in this environment.

## Ingestion Result

`security-rag ingest --source standards --db storage\chroma --batch-size 64`

Indexed chunks: `22798`

Chroma collection count after ingestion: `22801`, including the existing sample fixture chunks.
