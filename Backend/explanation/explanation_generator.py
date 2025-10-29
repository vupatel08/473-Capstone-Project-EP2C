"""
Explanation Generator
Creates bidirectional traceability maps between generated code and paper sections.
This module generates explanations, links, and documentation to make code explainable.
"""

import json
import re
from typing import Dict, List, Tuple, Optional
import openai
import os


class ExplanationGenerator:
    """
    Generates explainable documentation and traceability mappings
    between paper sections and generated code.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the Explanation Generator.
        
        Args:
            openai_api_key: OpenAI API key. If None, uses environment variable.
        """
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        openai.api_key = os.getenv("OPENAI_API_KEY")
    
    def generate_traceability_map(
        self,
        paper_json: Dict,
        generated_files: Dict[str, str],
        planning_artifacts: Dict
    ) -> Dict:
        """
        Create bidirectional links between code components and paper sections.
        
        Args:
            paper_json: Parsed paper content
            generated_files: Dictionary mapping file paths to code content
            planning_artifacts: Planning stage artifacts with file descriptions
            
        Returns:
            Dictionary containing bidirectional traceability maps
        """
        print("Generating paper-to-code traceability map...")
        
        # Extract paper structure
        paper_sections = self._extract_paper_sections(paper_json)
        
        # Build traceability map
        code_to_paper = {}
        paper_to_code = {}
        
        for file_path, code_content in generated_files.items():
            file_basename = os.path.basename(file_path)
            
            # Extract code components (classes, functions, methods)
            code_components = self._extract_code_components(code_content, file_path)
            
            # Map each component to paper sections
            for component, component_info in code_components.items():
                # Use planning artifacts to find related paper sections
                related_sections = self._find_related_paper_sections(
                    file_basename,
                    component_info,
                    planning_artifacts,
                    paper_sections
                )
                
                if related_sections:
                    code_to_paper[component] = related_sections
                    
                    # Build reverse map
                    for section in related_sections:
                        if section not in paper_to_code:
                            paper_to_code[section] = []
                        paper_to_code[section].append({
                            "component": component,
                            "description": component_info.get("description", ""),
                            "file": file_path
                        })
        
        return {
            "code_to_paper": code_to_paper,
            "paper_to_code": paper_to_code,
            "paper_sections": paper_sections,
            "coverage_score": self._calculate_coverage_score(code_to_paper, paper_sections)
        }
    
    def _extract_paper_sections(self, paper_json: Dict) -> List[Dict]:
        """Extract structured sections from paper JSON."""
        sections = []
        
        # Extract from paper body
        if "body_text" in paper_json:
            for item in paper_json["body_text"]:
                if "section" in item and "text" in item:
                    sections.append({
                        "section": item["section"],
                        "text": item["text"],
                        "ontology": item.get("section")
                    })
        
        # Extract from abstract if available
        if "abstract" in paper_json and paper_json["abstract"]:
            sections.append({
                "section": "Abstract",
                "text": paper_json["abstract"],
                "ontology": "abstract"
            })
        
        return sections
    
    def _extract_code_components(self, code_content: str, file_path: str) -> Dict:
        """Extract classes, functions, and methods from code."""
        components = {}
        
        # Extract classes
        class_pattern = r'class\s+(\w+)[^:]*:(.*?)(?=\nclass|\ndef\s|\Z)'
        for match in re.finditer(class_pattern, code_content, re.DOTALL):
            class_name = match.group(1)
            class_body = match.group(2)
            
            # Extract methods
            methods = []
            method_pattern = r'def\s+(\w+)\s*\([^)]*\):'
            for method_match in re.finditer(method_pattern, class_body):
                methods.append(method_match.group(1))
            
            component_key = f"{os.path.basename(file_path)}:{class_name}"
            components[component_key] = {
                "type": "class",
                "description": self._extract_docstring(class_body),
                "methods": methods,
                "file": file_path
            }
        
        # Extract standalone functions
        function_pattern = r'def\s+(\w+)\s*\([^)]*\):(.*?)(?=\ndef|\Z)'
        for match in re.finditer(function_pattern, code_content, re.DOTALL):
            func_name = match.group(1)
            func_body = match.group(2)
            
            component_key = f"{os.path.basename(file_path)}:{func_name}"
            components[component_key] = {
                "type": "function",
                "description": self._extract_docstring(func_body),
                "file": file_path
            }
        
        return components
    
    def _extract_docstring(self, code_block: str) -> str:
        """Extract docstring from code block."""
        # Try triple-quoted docstrings
        docstring_pattern = r'"""(.*?)"""'
        match = re.search(docstring_pattern, code_block, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Try single-quoted docstrings
        docstring_pattern = r"'''(.*?)'''"
        match = re.search(docstring_pattern, code_block, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        return ""
    
    def _find_related_paper_sections(
        self,
        file_name: str,
        component_info: Dict,
        planning_artifacts: Dict,
        paper_sections: List[Dict]
    ) -> List[str]:
        """Find paper sections related to a code component using LLM."""
        
        # Build context
        component_description = component_info.get("description", "")
        component_type = component_info.get("type", "")
        
        # Find file description from planning artifacts
        file_description = ""
        if "logic_analysis" in planning_artifacts:
            for item in planning_artifacts["logic_analysis"]:
                if isinstance(item, list) and len(item) > 0 and item[0] == file_name:
                    file_description = item[1] if len(item) > 1 else ""
        
        # Create prompt for LLM to find related sections
        prompt = f"""
Given the following code component and paper sections, identify which paper sections are most relevant.

Code Component:
- File: {file_name}
- Type: {component_type}
- Description: {component_description}
- File Purpose: {file_description}

Paper Sections (up to 10 most relevant):
{self._format_paper_sections_for_prompt(paper_sections[:10])}

Identify the 2-3 most relevant paper sections for this code component. 
Return only the section names/titles, separated by commas.
"""
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing code and linking it to paper sections."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0
            )
            
            sections = response.choices[0].message.content.strip().split(",")
            return [s.strip() for s in sections if s.strip()]
        except Exception as e:
            print(f"Error finding related sections: {e}")
            return []
    
    def _format_paper_sections_for_prompt(self, paper_sections: List[Dict]) -> str:
        """Format paper sections for LLM prompt."""
        formatted = []
        for i, section in enumerate(paper_sections[:10]):
            section_name = section.get("section", f"Section {i}")
            text_preview = section.get("text", "")[:200] + "..." if len(section.get("text", "")) > 200 else section.get("text", "")
            formatted.append(f"- {section_name}\n  {text_preview}")
        return "\n\n".join(formatted)
    
    def _calculate_coverage_score(self, code_to_paper: Dict, paper_sections: List[Dict]) -> float:
        """Calculate traceability coverage score."""
        if not paper_sections:
            return 0.0
        
        # Count unique sections that have code links
        linked_sections = set()
        for component, sections in code_to_paper.items():
            linked_sections.update(sections)
        
        coverage = len(linked_sections) / len(paper_sections) if paper_sections else 0.0
        return round(coverage, 3)
    
    def generate_explanation_summary(self, traceability_map: Dict) -> str:
        """Generate a human-readable explanation summary."""
        summary = "=== EXPLANATION LAYER SUMMARY ===\n\n"
        
        summary += f"Traceability Coverage: {traceability_map['coverage_score']*100:.1f}%\n"
        summary += f"Code Components Mapped: {len(traceability_map['code_to_paper'])}\n"
        summary += f"Paper Sections Covered: {len(traceability_map['paper_to_code'])}\n\n"
        
        summary += "Key Implementations:\n"
        for component, sections in list(traceability_map['code_to_paper'].items())[:5]:
            summary += f"- {component} → {', '.join(sections)}\n"
        
        return summary


if __name__ == "__main__":
    # Example usage
    generator = ExplanationGenerator()
    
    # Example data
    paper_json = {
        "abstract": "This paper presents...",
        "body_text": [
            {"section": "Introduction", "text": "..."},
            {"section": "Methodology", "text": "..."}
        ]
    }
    
    generated_files = {
        "model.py": "class Transformer(nn.Module):\n    def __init__(self):\n        pass",
        "trainer.py": "def train(model, data):\n    pass"
    }
    
    planning_artifacts = {
        "logic_analysis": [
            ["model.py", "Defines the Transformer architecture"],
            ["trainer.py", "Implements training loop"]
        ]
    }
    
    traceability_map = generator.generate_traceability_map(
        paper_json,
        generated_files,
        planning_artifacts
    )
    
    print(generator.generate_explanation_summary(traceability_map))

