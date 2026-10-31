"""Export services for packaging generated frontend code into deployable zip archives."""

import io
import zipfile
from typing import Optional


class ZipExportService:
    """Packages HTML/CSS/JS artifacts into standard compressed zip archives."""

    @staticmethod
    def export_to_zip(
        html_content: str,
        filename: str = "index.html",
        zip_filename: Optional[str] = None,
    ) -> bytes:
        """Package the generated HTML code string into an in-memory zip archive buffer.
        
        Args:
            html_content: Raw or sandboxed HTML/Tailwind/JS code string.
            filename: Name of the primary HTML file in the archive (defaults to 'index.html').
            zip_filename: Optional internal name or identifier.

        Returns:
            bytes: Binary content of the generated .zip archive.
        """
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            # Write primary HTML file encoded in UTF-8
            zf.writestr(filename, html_content.encode("utf-8"))
            
            # Write a minimal README file for immediate developer deployment
            readme_content = (
                "# Generated Workspace - RoboUI\n\n"
                "This workspace was synthesized on-device by RoboUI using local Gemma vision inference.\n\n"
                "## Getting Started\n"
                "Open `index.html` directly in any modern browser. All styles and scripts are completely self-contained.\n"
            )
            zf.writestr("README.md", readme_content.encode("utf-8"))

        return buffer.getvalue()
