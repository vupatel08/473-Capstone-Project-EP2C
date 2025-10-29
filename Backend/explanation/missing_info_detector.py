"""
Missing Information Detector
Identifies missing datasets, hyperparameters, and implementation details
that are not specified in the paper but needed for code execution.
"""

import re
from typing import Dict, List, Optional
import yaml


class MissingInfoDetector:
    """
    Detects missing information in generated code that should be in the paper.
    """
    
    # Common hyperparameters to check
    HYPERPARAMETERS = [
        "learning_rate",
        "batch_size",
        "epochs",
        "weight_decay",
        "dropout",
        "optimizer",
        "momentum",
        "beta1",
        "beta2"
    ]
    
    # Common dataset-related terms
    DATASET_TERMS = [
        "dataset",
        "data",
        "train",
        "test",
        "validation",
        "split"
    ]
    
    def __init__(self):
        """Initialize the Missing Information Detector."""
        pass
    
    def detect_missing_information(
        self,
        paper_content: str,
        generated_config: Dict,
        code_files: Dict[str, str]
    ) -> List[Dict]:
        """
        Detect missing information across multiple categories.
        
        Args:
            paper_content: Full paper text
            generated_config: Generated configuration dictionary
            code_files: Dictionary of generated code files
            
        Returns:
            List of missing information alerts
        """
        missing_info = []
        
        # Detect missing hyperparameters
        missing_info.extend(self.detect_missing_hyperparameters(paper_content, generated_config))
        
        # Detect missing dataset information
        missing_info.extend(self.detect_missing_dataset_info(paper_content, code_files))
        
        # Detect missing implementation details
        missing_info.extend(self.detect_missing_implementation_details(paper_content, code_files))
        
        # Detect missing hardware/performance specs
        missing_info.extend(self.detect_missing_performance_info(paper_content, code_files))
        
        return missing_info
    
    def detect_missing_hyperparameters(
        self,
        paper_content: str,
        config: Dict
    ) -> List[Dict]:
        """Identify hyperparameters not specified in paper."""
        missing = []
        paper_lower = paper_content.lower()
        
        for param in self.HYPERPARAMETERS:
            # Check if parameter is mentioned in paper
            param_mentioned = any(
                param in paper_lower or
                param.replace("_", " ") in paper_lower
            )
            
            # If parameter exists in config but not in paper
            if param in config and not param_mentioned:
                missing.append({
                    "type": "hyperparameter",
                    "parameter": param,
                    "description": f"Hyperparameter '{param}' is used in code but not explicitly specified in paper",
                    "current_value": str(config[param]),
                    "severity": self._determine_severity(param),
                    "suggestion": self._get_hyperparameter_suggestion(param)
                })
        
        return missing
    
    def detect_missing_dataset_info(
        self,
        paper_content: str,
        code_files: Dict[str, str]
    ) -> List[Dict]:
        """Identify missing dataset specifications."""
        missing = []
        
        # Check if dataset download/loading code exists
        has_dataset_loader = any(
            "load" in content.lower() or "dataset" in content.lower()
            for content in code_files.values()
        )
        
        if not has_dataset_loader:
            return missing
        
        # Check if dataset is mentioned in paper
        paper_lower = paper_content.lower()
        dataset_mentioned = any(term in paper_lower for term in self.DATASET_TERMS)
        
        if not dataset_mentioned:
            missing.append({
                "type": "dataset",
                "parameter": "dataset",
                "description": "Dataset loading code exists but dataset is not clearly specified in paper",
                "severity": "high",
                "suggestion": "Verify dataset compatibility with paper's experimental setup"
            })
        
        # Check for common dataset issues
        for file_path, content in code_files.items():
            # Check for hardcoded paths
            if re.search(r'["\']/[^"\']+["\']', content):
                missing.append({
                    "type": "dataset_path",
                    "parameter": "data_path",
                    "description": f"Hardcoded dataset path found in {file_path}",
                    "severity": "medium",
                    "suggestion": "Use config.yaml or environment variables for dataset paths"
                })
        
        return missing
    
    def detect_missing_implementation_details(
        self,
        paper_content: str,
        code_files: Dict[str, str]
    ) -> List[Dict]:
        """Identify missing implementation details."""
        missing = []
        
        for file_path, content in code_files.items():
            # Check for TODO or FIXME comments
            if re.search(r'TODO|FIXME', content, re.IGNORECASE):
                missing.append({
                    "type": "implementation",
                    "parameter": "todo",
                    "description": f"TODO/FIXME comments found in {file_path}",
                    "severity": "low",
                    "suggestion": "Review and complete TODO items"
                })
            
            # Check for placeholder values
            if re.search(r'PLACEHOLDER|XXX|TBD', content, re.IGNORECASE):
                missing.append({
                    "type": "implementation",
                    "parameter": "placeholder",
                    "description": f"Placeholder values found in {file_path}",
                    "severity": "medium",
                    "suggestion": "Replace placeholders with actual values"
                })
        
        return missing
    
    def detect_missing_performance_info(
        self,
        paper_content: str,
        code_files: Dict[str, str]
    ) -> List[Dict]:
        """Identify missing performance/hardware specifications."""
        missing = []
        
        paper_lower = paper_content.lower()
        
        # Check for GPU/CPU requirements
        has_gpu_code = any("cuda" in content.lower() or "gpu" in content.lower() 
                          for content in code_files.values())
        
        if has_gpu_code and "gpu" not in paper_lower and "cuda" not in paper_lower:
            missing.append({
                "type": "hardware",
                "parameter": "gpu",
                "description": "GPU/CUDA code found but hardware requirements not specified in paper",
                "severity": "low",
                "suggestion": "Verify GPU requirements match paper's experimental setup"
            })
        
        return missing
    
    def _determine_severity(self, parameter: str) -> str:
        """Determine severity level for missing parameter."""
        critical_params = ["learning_rate", "batch_size", "epochs"]
        if parameter in critical_params:
            return "high"
        elif parameter in ["dropout", "optimizer", "weight_decay"]:
            return "medium"
        else:
            return "low"
    
    def _get_hyperparameter_suggestion(self, parameter: str) -> str:
        """Get suggestion for missing hyperparameter."""
        suggestions = {
            "learning_rate": "Review paper's optimization section or experiment with typical values (0.001, 0.0001)",
            "batch_size": "Common values: 32, 64, 128. Consider memory constraints.",
            "epochs": "Typical range: 50-200. Monitor for overfitting.",
            "dropout": "Common values: 0.1, 0.2, 0.5. Used for regularization.",
            "optimizer": "Common choices: Adam, SGD, RMSprop",
            "weight_decay": "L2 regularization strength, typically 1e-4 to 1e-5",
            "momentum": "For SGD optimizer, typically 0.9",
            "beta1": "For Adam optimizer, typically 0.9",
            "beta2": "For Adam optimizer, typically 0.999"
        }
        return suggestions.get(parameter, "Review paper or standard practices")
    
    def generate_missing_info_summary(self, missing_info: List[Dict]) -> str:
        """Generate human-readable summary of missing information."""
        if not missing_info:
            return "✅ No missing information detected."
        
        summary = f"⚠️ Found {len(missing_info)} items requiring attention:\n\n"
        
        # Group by severity
        by_severity = {"high": [], "medium": [], "low": []}
        for item in missing_info:
            severity = item.get("severity", "low")
            by_severity[severity].append(item)
        
        for severity in ["high", "medium", "low"]:
            if by_severity[severity]:
                summary += f"\n{severity.upper()}: {len(by_severity[severity])} items\n"
                for item in by_severity[severity][:3]:  # Show first 3
                    summary += f"  - {item['parameter']}: {item['description']}\n"
        
        return summary


if __name__ == "__main__":
    # Example usage
    detector = MissingInfoDetector()
    
    paper_content = """
    We present a novel architecture for sequence modeling. Our model uses 
    multi-head attention mechanism.
    """
    
    config = {
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 100
    }
    
    code_files = {
        "trainer.py": "def train():\n    lr = config['learning_rate']"
    }
    
    missing = detector.detect_missing_information(paper_content, config, code_files)
    print(detector.generate_missing_info_summary(missing))

