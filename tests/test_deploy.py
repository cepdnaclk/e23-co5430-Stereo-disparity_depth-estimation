"""Tests for deployment scripts and directory organization.

Run with:
    python3 -m unittest tests/test_deploy.py
"""

import os
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEPLOY_DIR = REPO_ROOT / "deploy"


class DeploymentStructureTests(unittest.TestCase):
    def test_deploy_files_exist_in_deploy_folder(self):
        """All deployment scripts must exist in the deploy/ folder."""
        expected_files = [
            "deploy_to_hf.sh",
            "deploy_to_hf.bat",
            "deploy_to_hf.ps1",
        ]
        for filename in expected_files:
            file_path = DEPLOY_DIR / filename
            self.assertTrue(file_path.is_file(), f"{filename} missing in {DEPLOY_DIR}")

    def test_no_deploy_files_in_root(self):
        """Old deployment scripts must not remain in repository root."""
        obsolete_files = [
            "deploy_to_hf.sh",
            "deploy_to_hf.bat",
            "deploy_to_hf.ps1",
        ]
        for filename in obsolete_files:
            file_path = REPO_ROOT / filename
            self.assertFalse(
                file_path.exists(),
                f"{filename} should not exist in repo root; must be in deploy/",
            )

    def test_deploy_sh_executable(self):
        """deploy_to_hf.sh must have executable permissions."""
        script_path = DEPLOY_DIR / "deploy_to_hf.sh"
        self.assertTrue(
            os.access(script_path, os.X_OK),
            "deploy_to_hf.sh is not marked as executable",
        )

    def test_deploy_sh_dry_run_from_root(self):
        """Running deploy_to_hf.sh --dry-run from repository root must succeed and produce clean package."""
        proc = subprocess.run(
            ["./deploy/deploy_to_hf.sh", "--dry-run"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"deploy_to_hf.sh failed with output:\n{proc.stdout}\n{proc.stderr}",
        )
        self.assertIn("Dry run passed! Clean package verified successfully.", proc.stdout)

        # Ensure essential code is packaged
        for expected in ("app.py", "README.md", "requirements.txt", "src/dataset.py", "core/raft_stereo.py"):
            self.assertIn(f"  - {expected}", proc.stdout)

        # Ensure demo images, documents, deploy scripts, and packages.txt are NOT staged into Space package
        for excluded in ("deploy/", "demo_images/", "Documents/", "report_figures/", "packages.txt"):
            self.assertNotIn(f"  - {excluded}", proc.stdout)

    def test_deploy_sh_dry_run_from_deploy_dir(self):
        """Running deploy_to_hf.sh from inside deploy/ must navigate to root and succeed."""
        proc = subprocess.run(
            ["./deploy_to_hf.sh", "--dry-run"],
            cwd=str(DEPLOY_DIR),
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"deploy_to_hf.sh from deploy/ failed:\n{proc.stdout}\n{proc.stderr}",
        )
        self.assertIn("Dry run passed! Clean package verified successfully.", proc.stdout)
        self.assertIn("  - app.py", proc.stdout)

    def test_windows_scripts_repo_root_resolution(self):
        """Batch and PowerShell scripts must contain repo root navigation logic."""
        bat_content = (DEPLOY_DIR / "deploy_to_hf.bat").read_text(encoding="utf-8")
        self.assertIn('cd /d "%~dp0.."', bat_content)

        ps1_content = (DEPLOY_DIR / "deploy_to_hf.ps1").read_text(encoding="utf-8")
        self.assertIn("Split-Path -Parent $ScriptDir", ps1_content)
        self.assertIn(".\\deploy\\deploy_to_hf.ps1", ps1_content)

    def test_documentation_and_reports_reference_deploy_folder(self):
        """README.md and generate_ieee_report.py must reference the new deploy/ paths."""
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("./deploy/deploy_to_hf.sh", readme)
        self.assertIn(r"deploy\deploy_to_hf.bat", readme)
        self.assertIn(r".\deploy\deploy_to_hf.ps1", readme)
        self.assertIn("├── deploy/", readme)

        report_gen = (REPO_ROOT / "generate_ieee_report.py").read_text(encoding="utf-8")
        self.assertIn("deploy/deploy_to_hf.sh", report_gen)
        self.assertNotIn('"deploy_to_hf.sh"', report_gen)


if __name__ == "__main__":
    unittest.main()
