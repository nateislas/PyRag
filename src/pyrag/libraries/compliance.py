"""Compliance tracking for legal compliance and licensing."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class ComplianceStatus:
    """Compliance status for a library."""

    library_name: str
    license_type: str
    license_approved: bool
    maintainer_contacted: bool
    opt_out_status: str  # 'none', 'requested', 'granted', 'denied'
    last_updated: datetime
    compliance_score: float


class ComplianceTracker:
    """Track legal compliance and licensing for libraries."""

    def __init__(self):
        """Initialize the compliance tracker."""
        self.logger = get_logger(__name__)

        # Track compliance status
        self.compliance_records: Dict[str, ComplianceStatus] = {}

        # Approved license types
        self.approved_licenses = {
            "MIT",
            "Apache-2.0",
            "BSD-3-Clause",
            "BSD-2-Clause",
            "ISC",
            "Unlicense",
            "CC0-1.0",
            "MPL-2.0",
        }

        # License type mappings
        self.license_mappings = {
            "mit": "MIT",
            "apache": "Apache-2.0",
            "apache-2.0": "Apache-2.0",
            "bsd": "BSD-3-Clause",
            "bsd-3-clause": "BSD-3-Clause",
            "bsd-2-clause": "BSD-2-Clause",
            "isc": "ISC",
            "unlicense": "Unlicense",
            "cc0": "CC0-1.0",
            "mpl": "MPL-2.0",
            "mpl-2.0": "MPL-2.0",
        }

    async def check_library_compliance(
        self, library_info: Dict[str, Any]
    ) -> ComplianceStatus:
        """Check compliance status for a library."""
        try:
            library_name = library_info.get("name", "unknown")
            self.logger.info(f"Checking compliance for library: {library_name}")

            # Get license information
            license_info = library_info.get("license", "")
            license_type = self._normalize_license(license_info)
            license_approved = license_type in self.approved_licenses

            # Check if we have existing compliance record
            existing_record = self.compliance_records.get(library_name)

            if existing_record:
                # Update existing record
                existing_record.license_type = license_type
                existing_record.license_approved = license_approved
                existing_record.last_updated = datetime.now()
                existing_record.compliance_score = self._calculate_compliance_score(
                    existing_record
                )

                return existing_record
            else:
                # Create new compliance record
                compliance_status = ComplianceStatus(
                    library_name=library_name,
                    license_type=license_type,
                    license_approved=license_approved,
                    maintainer_contacted=False,
                    opt_out_status="none",
                    last_updated=datetime.now(),
                    compliance_score=0.0,
                )

                # Calculate compliance score
                compliance_status.compliance_score = self._calculate_compliance_score(
                    compliance_status
                )

                # Store record
                self.compliance_records[library_name] = compliance_status

                self.logger.info(f"Created compliance record for {library_name}")
                return compliance_status

        except Exception as e:
            self.logger.error(
                f"Failed to check compliance for {library_info.get('name', 'unknown')}: {e}"
            )
            return ComplianceStatus(
                library_name=library_info.get("name", "unknown"),
                license_type="unknown",
                license_approved=False,
                maintainer_contacted=False,
                opt_out_status="none",
                last_updated=datetime.now(),
                compliance_score=0.0,
            )

    def _normalize_license(self, license_info: str) -> str:
        """Normalize license information to standard format."""
        if not license_info:
            return "unknown"

        # Convert to lowercase for comparison
        license_lower = license_info.lower()

        # Check for exact matches in mappings
        for key, value in self.license_mappings.items():
            if key in license_lower:
                return value

        # Check for partial matches
        if "mit" in license_lower:
            return "MIT"
        elif "apache" in license_lower:
            return "Apache-2.0"
        elif "bsd" in license_lower:
            return "BSD-3-Clause"
        elif "gpl" in license_lower:
            return "GPL"  # Not in approved list
        elif "lgpl" in license_lower:
            return "LGPL"  # Not in approved list
        else:
            return license_info  # Return original if no match

    def _calculate_compliance_score(self, compliance_status: ComplianceStatus) -> float:
        """Calculate compliance score based on various factors."""
        score = 0.0

        # License approval (50% weight)
        if compliance_status.license_approved:
            score += 0.5

        # Maintainer contact status (20% weight)
        if compliance_status.maintainer_contacted:
            score += 0.2

        # Opt-out status (30% weight)
        if compliance_status.opt_out_status == "none":
            score += 0.3
        elif compliance_status.opt_out_status == "denied":
            score += 0.3
        elif compliance_status.opt_out_status == "requested":
            score += 0.1
        # "granted" gets 0 points

        return min(score, 1.0)

    async def contact_maintainer(
        self, library_name: str, contact_info: Dict[str, Any]
    ) -> bool:
        """Contact library maintainer for permission."""
        try:
            self.logger.info(f"Contacting maintainer for library: {library_name}")

            # Get compliance record
            compliance_record = self.compliance_records.get(library_name)
            if not compliance_record:
                self.logger.warning(f"No compliance record found for {library_name}")
                return False

            # Update maintainer contact status
            compliance_record.maintainer_contacted = True
            compliance_record.last_updated = datetime.now()
            compliance_record.compliance_score = self._calculate_compliance_score(
                compliance_record
            )

            # In production, this would send an actual email/notification
            self.logger.info(f"Maintainer contact recorded for {library_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to contact maintainer for {library_name}: {e}")
            return False

    async def request_opt_out(self, library_name: str, reason: str) -> bool:
        """Request opt-out for a library."""
        try:
            self.logger.info(f"Requesting opt-out for library: {library_name}")

            # Get compliance record
            compliance_record = self.compliance_records.get(library_name)
            if not compliance_record:
                self.logger.warning(f"No compliance record found for {library_name}")
                return False

            # Update opt-out status
            compliance_record.opt_out_status = "requested"
            compliance_record.last_updated = datetime.now()
            compliance_record.compliance_score = self._calculate_compliance_score(
                compliance_record
            )

            # In production, this would send an opt-out request
            self.logger.info(f"Opt-out request recorded for {library_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to request opt-out for {library_name}: {e}")
            return False

    async def grant_opt_out(self, library_name: str) -> bool:
        """Grant opt-out request for a library."""
        try:
            self.logger.info(f"Granting opt-out for library: {library_name}")

            # Get compliance record
            compliance_record = self.compliance_records.get(library_name)
            if not compliance_record:
                self.logger.warning(f"No compliance record found for {library_name}")
                return False

            # Update opt-out status
            compliance_record.opt_out_status = "granted"
            compliance_record.last_updated = datetime.now()
            compliance_record.compliance_score = self._calculate_compliance_score(
                compliance_record
            )

            self.logger.info(f"Opt-out granted for {library_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to grant opt-out for {library_name}: {e}")
            return False

    async def deny_opt_out(self, library_name: str, reason: str) -> bool:
        """Deny opt-out request for a library."""
        try:
            self.logger.info(f"Denying opt-out for library: {library_name}")

            # Get compliance record
            compliance_record = self.compliance_records.get(library_name)
            if not compliance_record:
                self.logger.warning(f"No compliance record found for {library_name}")
                return False

            # Update opt-out status
            compliance_record.opt_out_status = "denied"
            compliance_record.last_updated = datetime.now()
            compliance_record.compliance_score = self._calculate_compliance_score(
                compliance_record
            )

            self.logger.info(f"Opt-out denied for {library_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to deny opt-out for {library_name}: {e}")
            return False

    async def get_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report for all libraries."""
        try:
            total_libraries = len(self.compliance_records)

            if total_libraries == 0:
                return {
                    "total_libraries": 0,
                    "compliant_libraries": 0,
                    "non_compliant_libraries": 0,
                    "pending_opt_outs": 0,
                    "average_compliance_score": 0.0,
                    "libraries": [],
                }

            # Calculate statistics
            compliant_libraries = sum(
                1
                for record in self.compliance_records.values()
                if record.compliance_score >= 0.7
            )
            non_compliant_libraries = sum(
                1
                for record in self.compliance_records.values()
                if record.compliance_score < 0.7
            )
            pending_opt_outs = sum(
                1
                for record in self.compliance_records.values()
                if record.opt_out_status == "requested"
            )

            avg_compliance_score = (
                sum(
                    record.compliance_score
                    for record in self.compliance_records.values()
                )
                / total_libraries
            )

            # Get library details
            libraries = []
            for record in self.compliance_records.values():
                libraries.append(
                    {
                        "name": record.library_name,
                        "license_type": record.license_type,
                        "license_approved": record.license_approved,
                        "maintainer_contacted": record.maintainer_contacted,
                        "opt_out_status": record.opt_out_status,
                        "compliance_score": record.compliance_score,
                        "last_updated": record.last_updated.isoformat(),
                    }
                )

            return {
                "total_libraries": total_libraries,
                "compliant_libraries": compliant_libraries,
                "non_compliant_libraries": non_compliant_libraries,
                "pending_opt_outs": pending_opt_outs,
                "average_compliance_score": avg_compliance_score,
                "libraries": libraries,
            }

        except Exception as e:
            self.logger.error(f"Failed to generate compliance report: {e}")
            return {
                "error": str(e),
                "total_libraries": 0,
                "compliant_libraries": 0,
                "non_compliant_libraries": 0,
                "pending_opt_outs": 0,
                "average_compliance_score": 0.0,
                "libraries": [],
            }

    async def get_library_compliance(
        self, library_name: str
    ) -> Optional[ComplianceStatus]:
        """Get compliance status for a specific library."""
        return self.compliance_records.get(library_name)

    async def update_license_info(self, library_name: str, new_license: str) -> bool:
        """Update license information for a library."""
        try:
            self.logger.info(f"Updating license info for {library_name}: {new_license}")

            # Get compliance record
            compliance_record = self.compliance_records.get(library_name)
            if not compliance_record:
                self.logger.warning(f"No compliance record found for {library_name}")
                return False

            # Update license information
            normalized_license = self._normalize_license(new_license)
            compliance_record.license_type = normalized_license
            compliance_record.license_approved = (
                normalized_license in self.approved_licenses
            )
            compliance_record.last_updated = datetime.now()
            compliance_record.compliance_score = self._calculate_compliance_score(
                compliance_record
            )

            self.logger.info(f"License info updated for {library_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to update license info for {library_name}: {e}")
            return False
