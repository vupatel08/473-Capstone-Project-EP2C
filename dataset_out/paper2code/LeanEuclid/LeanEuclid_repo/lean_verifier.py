## lean_verifier.py
import subprocess
import tempfile
import os
import logging
from typing import List, Optional

class LeanVerifier:
    """
    Class to verify a given proof script in Lean environment.
    Uses subprocess to invoke local Lean compiler or checker.
    """
    def __init__(self, lean_path: str = "/usr/bin/lean", timeout: int = 10):
        """
        Initialize the verifier with path to Lean executable and timeout.
        """
        self.lean_path: str = lean_path
        self.timeout: int = timeout
        # Set up logging
        logging.basicConfig(level=logging.INFO)

    def verify_proof(self, tactics: List[str], theorem_statement: str = "") -> bool:
        """
        Verify the proof tactics sequence against the theorem statement.
        Returns True if proof is successfully verified, False otherwise.

        Args:
            tactics (List[str]): List of tactic commands in string form.
            theorem_statement (str): The theorem statement (not strictly needed, but for context).

        Returns:
            bool: True if the proof verifies successfully, False otherwise.
        """
        # Build the full proof script
        proof_script = self._construct_proof_script(tactics, theorem_statement)

        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lean', delete=False) as tf:
            filename = tf.name
            tf.write(proof_script)
        try:
            # Call Lean to check the proof
            cmd = [self.lean_path, '--check', filename]
            # Run the process
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout,
                universal_newlines=True
            )
            # Debug logs
            logging.info(f"Running command: {' '.join(cmd)}")
            logging.info(f"Lean stdout: {result.stdout}")
            logging.info(f"Lean stderr: {result.stderr}")

            # Check success: lean returns 0 exit code if proof verified
            if result.returncode == 0:
                return True
            else:
                # Verification failed; log error message
                logging.warning(f"Proof verification failed for {filename}")
                return False
        except subprocess.TimeoutExpired:
            logging.error("Lean verification timed out.")
            return False
        except Exception as e:
            logging.exception(f"Error during Lean verification: {e}")
            return False
        finally:
            # Clean up the temp file
            try:
                os.remove(filename)
            except OSError:
                pass

    def _construct_proof_script(self, tactics: List[str], theorem_statement: str) -> str:
        """
        Compose the full Lean proof script with import statements, theorem statement,
        and tactics sequence.
        """
        # Basic Lean import and environment setup, assuming standard import
        header = "import Euclid\n\n"
        # Optional: include the theorem statement as a comment or as the theorem declaration
        # Here, we create a dummy theorem with the provided statement for verification
        # or assume the tactics are embedded within a theorem declaration
        # For this implementation, assume tactics are within a theorem named 'proven_theorem'
        theorem_declaration = f"theorem proven_theorem : \n"
        # For verifying tactics, wrap them into a definition/proof block
        proof_block = "begin\n"
        for t in tactics:
            # Each tactic line is appended
            proof_block += f"  {t.strip()}\n"
        proof_block += "end\n"

        # Alternatively, if tactics are provided as a sequence, place directly
        # For safety, wrap in a proof block
        script = header
        script += f"theorem temp_verification : {theorem_statement}\n"
        script += "begin\n"
        for t in tactics:
            script += f"  {t}\n"
        script += "end\n"
        return script
