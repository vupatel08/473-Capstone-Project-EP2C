"""
Explainability Pipeline
Main integration script that orchestrates the explanation layer generation.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
backend_dir = Path(__file__).parent.parent.resolve()
project_root = backend_dir.parent
env_paths = [backend_dir / ".env", project_root / ".env"]
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        break
else:
    load_dotenv()

from explanation_generator import ExplanationGenerator
from readme_generator import READMEGenerator
from missing_info_detector import MissingInfoDetector
from explanation_evaluator import ExplanationEvaluator


class ExplainabilityPipeline:
    """
    Main pipeline for generating comprehensive explainable documentation.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the explainability pipeline."""
        self.explanation_generator = ExplanationGenerator(openai_api_key)
        self.readme_generator = READMEGenerator()
        self.missing_info_detector = MissingInfoDetector()
        self.evaluator = ExplanationEvaluator()
    
    def generate_explanation_layer(
        self,
        paper_json_path: str,
        generated_code_dir: str,
        planning_artifacts_path: str,
        output_dir: str,
        config_path: Optional[str] = None
    ) -> Dict:
        """
        Generate complete explanation layer for the generated repository.
        
        Args:
            paper_json_path: Path to parsed paper JSON
            generated_code_dir: Directory containing generated code files
            planning_artifacts_path: Path to planning artifacts JSON
            output_dir: Output directory for explanation layer files
            config_path: Path to config.yaml (optional)
            
        Returns:
            Dictionary containing all explanation layer outputs
        """
        print("\n" + "="*60)
        print("EP2C EXPLANATION LAYER GENERATION")
        print("="*60 + "\n")
        
        # Load all required inputs for explanation generation
        print("Loading inputs...")
        paper_json = self._load_json(paper_json_path)  # Paper content in JSON format
        generated_files = self._load_generated_files(generated_code_dir)  # All generated Python files
        planning_artifacts = self._load_json(planning_artifacts_path)  # Planning phase outputs
        config_data = self._load_yaml(config_path) if config_path else {}  # Config if available
        
        # Extract paper content as string for text-based analysis (missing info detection)
        paper_content = self._extract_paper_content(paper_json)
        
        # Step 1: Generate bidirectional traceability map (code <-> paper sections)
        print("\nStep 1: Generating paper-to-code traceability map...")
        traceability_map = self.explanation_generator.generate_traceability_map(
            paper_json,
            generated_files,
            planning_artifacts
        )
        
        # Step 2: Detect missing information (hyperparameters, datasets, etc. not in paper)
        print("\nStep 2: Detecting missing information...")
        missing_info = self.missing_info_detector.detect_missing_information(
            paper_content,
            config_data,
            generated_files
        )
        
        # Step 3: Generate basic README with traceability and missing info alerts
        print("\nStep 3: Generating comprehensive README...")
        readme_content = self.readme_generator.generate_readme(
            paper_metadata=self._extract_paper_metadata(paper_json),
            code_structure=planning_artifacts,
            traceability_map=traceability_map,
            missing_info=missing_info,
            config_data=config_data
        )
        
        # Step 4: Evaluate explainability metrics (coverage, comments, references, etc.)
        print("\nStep 4: Evaluating explainability metrics...")
        explainability_metrics = self.evaluator.evaluate_explainability(
            generated_code=generated_files,
            traceability_map=traceability_map,
            missing_info=missing_info,
            paper_sections=traceability_map.get("paper_sections", [])
        )
        
        # Create output directory for all explanation layer files
        os.makedirs(output_dir, exist_ok=True)
        
        # Save all explanation layer outputs
        print("\nSaving explanation layer outputs...")
        
        # Save traceability map (JSON format for programmatic access)
        with open(f"{output_dir}/traceability_map.json", 'w', encoding='utf-8') as f:
            json.dump(traceability_map, f, indent=2)
        
        # Save basic README (user-facing documentation)
        with open(f"{output_dir}/README.md", 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        # Save missing information alerts (JSON format)
        with open(f"{output_dir}/missing_information.json", 'w', encoding='utf-8') as f:
            json.dump(missing_info, f, indent=2)
        
        # Save explainability metrics (JSON format)
        with open(f"{output_dir}/explainability_metrics.json", 'w', encoding='utf-8') as f:
            json.dump(explainability_metrics, f, indent=2)
        
        # Save human-readable evaluation report
        report = self.evaluator.generate_explanation_report(explainability_metrics)
        with open(f"{output_dir}/explainability_report.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Step 5: Generate comprehensive EXPLANATION.md (combines all outputs)
        print("\nStep 5: Generating comprehensive EXPLANATION.md...")
        comprehensive_explanation = self.readme_generator.generate_comprehensive_explanation(
            paper_metadata=self._extract_paper_metadata(paper_json),
            code_structure=planning_artifacts,
            traceability_map=traceability_map,
            missing_info=missing_info,
            metrics=explainability_metrics,
            config_data=config_data
        )
        with open(f"{output_dir}/EXPLANATION.md", 'w', encoding='utf-8') as f:
            f.write(comprehensive_explanation)
        
        print("\n" + "="*60)
        print("EXPLANATION LAYER GENERATION COMPLETE")
        print("="*60)
        print(f"\nOverall Explainability Score: {explainability_metrics['overall_explainability_score']:.2%}")
        print(f"Files saved to: {output_dir}")
        print(f"   - EXPLANATION.md (comprehensive documentation)")
        print(f"   - README.md (basic documentation)")
        print(f"   - traceability_map.json")
        print(f"   - missing_information.json")
        print(f"   - explainability_metrics.json")
        print(f"   - explainability_report.txt")
        print("="*60 + "\n")
        
        return {
            "traceability_map": traceability_map,
            "missing_info": missing_info,
            "readme": readme_content,
            "comprehensive_explanation": comprehensive_explanation,
            "metrics": explainability_metrics,
            "report": report
        }
    
    def _load_json(self, path: str) -> Dict:
        """Load JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_yaml(self, path: str) -> Dict:
        """Load YAML file."""
        import yaml
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_generated_files(self, code_dir: str) -> Dict[str, str]:
        """Load generated code files from directory."""
        files = {}
        
        # Walk through code directory and find all Python files
        for root, dirs, filenames in os.walk(code_dir):
            for filename in filenames:
                if filename.endswith('.py'):  # Only load Python files
                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, code_dir)  # Relative path for mapping
                    with open(file_path, 'r', encoding='utf-8') as f:
                        files[rel_path] = f.read()  # Store file content
        
        return files
    
    def _extract_paper_content(self, paper_json: Dict) -> str:
        """Extract paper content as string for text-based analysis."""
        content_parts = []
        
        # Add abstract if available
        if "abstract" in paper_json:
            content_parts.append(paper_json["abstract"])
        
        # Add all body text sections
        if "body_text" in paper_json:
            for item in paper_json["body_text"]:
                if "text" in item:
                    content_parts.append(item["text"])
        
        # Join all content into single string for text search
        return " ".join(content_parts)
    
    def _extract_paper_metadata(self, paper_json: Dict) -> Dict:
        """Extract paper metadata."""
        return {
            "title": paper_json.get("title", "Unknown"),
            "authors": paper_json.get("authors", []),
            "url": paper_json.get("url", ""),
            "abstract": paper_json.get("abstract", "")
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate explanation layer for generated code")
    parser.add_argument("--paper_json", type=str, required=True, help="Path to paper JSON")
    parser.add_argument("--code_dir", type=str, required=True, help="Directory with generated code")
    parser.add_argument("--planning_artifacts", type=str, required=True, help="Path to planning artifacts")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--config", type=str, help="Path to config.yaml (optional)")
    
    args = parser.parse_args()
    
    pipeline = ExplainabilityPipeline()
    pipeline.generate_explanation_layer(
        args.paper_json,
        args.code_dir,
        args.planning_artifacts,
        args.output_dir,
        args.config
    )

