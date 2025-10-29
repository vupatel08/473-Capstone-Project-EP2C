"""
Explanation Evaluator
Evaluates the quality of explanations, traceability, and overall explainability
of the generated code repository.
"""

import re
from typing import Dict, List
import json


class ExplanationEvaluator:
    """
    Evaluates explainability metrics for generated code.
    """
    
    def __init__(self):
        """Initialize the Explanation Evaluator."""
        pass
    
    def evaluate_explainability(
        self,
        generated_code: Dict[str, str],
        traceability_map: Dict,
        missing_info: List[Dict],
        paper_sections: List[Dict]
    ) -> Dict:
        """
        Evaluate overall explainability quality.
        
        Args:
            generated_code: Dictionary of generated code files
            traceability_map: Code-to-paper traceability map
            missing_info: List of missing information alerts
            paper_sections: List of paper sections
            
        Returns:
            Dictionary of explainability metrics
        """
        metrics = {
            "traceability_coverage": self.calculate_traceability_coverage(
                traceability_map, paper_sections
            ),
            "comment_density": self.calculate_comment_density(generated_code),
            "paper_reference_accuracy": self.calculate_paper_reference_accuracy(
                generated_code
            ),
            "missing_info_score": self.calculate_missing_info_score(missing_info),
            "readability_score": self.calculate_readability_score(generated_code),
            "overall_explainability_score": 0.0
        }
        
        # Calculate overall score (weighted average)
        weights = {
            "traceability_coverage": 0.3,
            "comment_density": 0.2,
            "paper_reference_accuracy": 0.25,
            "missing_info_score": 0.15,
            "readability_score": 0.1
        }
        
        metrics["overall_explainability_score"] = sum(
            metrics[key] * weights[key] for key in weights.keys()
        )
        
        return metrics
    
    def calculate_traceability_coverage(
        self,
        traceability_map: Dict,
        paper_sections: List[Dict]
    ) -> float:
        """Calculate how much of the paper is covered by code."""
        if "coverage_score" in traceability_map:
            return traceability_map["coverage_score"]
        return 0.0
    
    def calculate_comment_density(self, generated_code: Dict[str, str]) -> float:
        """Calculate comment density in generated code."""
        if not generated_code:
            return 0.0
        
        total_lines = 0
        comment_lines = 0
        
        for file_path, content in generated_code.items():
            lines = content.split('\n')
            total_lines += len(lines)
            
            for line in lines:
                stripped = line.strip()
                # Count comment lines
                if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                    comment_lines += 1
        
        if total_lines == 0:
            return 0.0
        
        return min(comment_lines / total_lines, 1.0)
    
    def calculate_paper_reference_accuracy(self, generated_code: Dict[str, str]) -> float:
        """Calculate how well code references paper sections."""
        if not generated_code:
            return 0.0
        
        total_files = len(generated_code)
        files_with_references = 0
        
        paper_reference_patterns = [
            r'section\s+\d+',
            r'equation\s*\(\d+\)',
            r'figure\s+\d+',
            r'table\s+\d+',
            r'algorithm\s+\d+',
            r'described\s+in',
            r'implements\s+the',
            r'according\s+to\s+the\s+paper'
        ]
        
        for file_path, content in generated_code.items():
            content_lower = content.lower()
            has_reference = any(
                re.search(pattern, content_lower, re.IGNORECASE)
                for pattern in paper_reference_patterns
            )
            
            if has_reference:
                files_with_references += 1
        
        if total_files == 0:
            return 0.0
        
        return files_with_references / total_files
    
    def calculate_missing_info_score(self, missing_info: List[Dict]) -> float:
        """Calculate score based on missing information (lower is better, inverted)."""
        if not missing_info:
            return 1.0
        
        # Weight by severity
        severity_weights = {"high": 3, "medium": 2, "low": 1}
        
        total_weight = sum(
            severity_weights.get(item.get("severity", "low"), 1)
            for item in missing_info
        )
        
        # Normalize (assume max possible score is 10 items * 3 weight = 30)
        normalized_score = 1.0 - min(total_weight / 30.0, 1.0)
        
        return max(normalized_score, 0.0)
    
    def calculate_readability_score(self, generated_code: Dict[str, str]) -> float:
        """Calculate code readability score."""
        if not generated_code:
            return 0.0
        
        scores = []
        
        for file_path, content in generated_code.items():
            file_score = 0.0
            
            # Check for docstrings
            if re.search(r'""".*?"""', content, re.DOTALL) or re.search(r"'''.*?'''", content, re.DOTALL):
                file_score += 0.3
            
            # Check for function/class docstrings
            functions_with_docs = len(re.findall(r'def\s+\w+\s*\([^)]*\):(.*?)("""|\'\'\')', content, re.DOTALL))
            total_functions = len(re.findall(r'def\s+\w+\s*\(', content))
            
            if total_functions > 0:
                file_score += 0.3 * (functions_with_docs / total_functions)
            
            # Check for type hints
            has_type_hints = bool(re.search(r':\s*\w+\s*[=,]', content))
            if has_type_hints:
                file_score += 0.2
            
            # Check for inline comments
            lines = content.split('\n')
            comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
            total_lines = len(lines)
            
            if total_lines > 0:
                file_score += 0.2 * min(comment_lines / total_lines * 5, 1.0)
            
            scores.append(min(file_score, 1.0))
        
        if not scores:
            return 0.0
        
        return sum(scores) / len(scores)
    
    def generate_explanation_report(self, metrics: Dict) -> str:
        """Generate human-readable explanation evaluation report."""
        report = "=" * 60 + "\n"
        report += "EXPLAINABILITY EVALUATION REPORT\n"
        report += "=" * 60 + "\n\n"
        
        report += f"Overall Explainability Score: {metrics['overall_explainability_score']:.2%}\n\n"
        
        report += "Detailed Metrics:\n"
        report += f"  • Traceability Coverage:    {metrics['traceability_coverage']:.2%}\n"
        report += f"  • Comment Density:          {metrics['comment_density']:.2%}\n"
        report += f"  • Paper Reference Accuracy: {metrics['paper_reference_accuracy']:.2%}\n"
        report += f"  • Missing Info Score:       {metrics['missing_info_score']:.2%}\n"
        report += f"  • Readability Score:        {metrics['readability_score']:.2%}\n\n"
        
        # Interpretation
        overall = metrics['overall_explainability_score']
        if overall >= 0.8:
            interpretation = "Excellent explainability - Code is well-documented and traceable"
        elif overall >= 0.6:
            interpretation = "Good explainability - Servlet overall explanation quality present"
        elif overall >= 0.4:
            interpretation = "Fair explainability - Some areas need improvement"
        else:
            interpretation = "Low explainability - Significant improvements needed"
        
        report += f"Interpretation: {interpretation}\n\n"
        
        # Recommendations
        report += "Recommendations:\n"
        if metrics['traceability_coverage'] < 0.5:
            report += "  • Improve traceability coverage by adding more code-paper links\n"
        if metrics['comment_density'] < 0.2:
            report += "  • Add more comments and docstrings to explain code logic\n"
        if metrics['paper_reference_accuracy'] < 0.5:
            report += "  • Add explicit references to paper sections, equations, and figures\n"
        if metrics['missing_info_score'] < 0.7:
            report += "  • Document missing information to help users understand gaps\n"
        if metrics['readability_score'] < 0.5:
            report += "  • Improve code readability with better structure and documentation\n"
        
        report += "=" * 60 + "\n"
        
        return report


if __name__ == "__main__":
    # Example usage
    evaluator = ExplanationEvaluator()
    
    generated_code = {
        "model.py": '''
"""Model implementation for paper."""
class Model:
    """Implements Section 3.2 of the paper."""
    def forward(self, x):
        # Equation (1) implementation
        return x
''',
        "trainer.py": "def train():\n    pass"
    }
    
    traceability_map = {
        "code_to_paper": {
            "model.py:Model": ["Section 3.2"]
        },
        "coverage_score": 0.6
    }
    
    missing_info = [
        {"severity": "high", "parameter": "learning_rate"},
        {"severity": "medium", "parameter": "dataset"}
    ]
    
    metrics = evaluator.evaluate_explainability(
        generated_code,
        traceability_map,
        missing_info,
        [{"section": "Section 1"}, {"section": "Section 2"}]
    )
    
    print(evaluator.generate_explanation_report(metrics))

