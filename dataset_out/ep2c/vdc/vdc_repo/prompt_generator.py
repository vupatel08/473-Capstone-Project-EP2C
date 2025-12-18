## prompt_generator.py
import random
from typing import List

class PromptGenerator:
    """
    Implements question prompt generation for the VDC pipeline's question modules.
    Generates both general and label-specific questions using structured templates.
    """

    def __init__(self, dataset_name: str = 'generic'):
        """
        Initialize the PromptGenerator with dataset context if necessary.
        Supports custom prompts per dataset type.
        Args:
            dataset_name (str): Optional dataset identifier for customization.
        """
        self.dataset_name = dataset_name
        # Predefine set of templates for general questions (as per Appendix E1)
        self.general_templates: List[str] = [
            "Describe the image in detail.",
            "Describe the image briefly.",
            "How would you summarize the content of the image in a few words?",
            "Provide a detailed description of the given image.",
            "Describe the image concisely.",
            "Provide a brief description of the given image.",
            "Offer a succinct explanation of the picture presented.",
            "Summarize the visual content of the image.",
            "Give a short and clear explanation of the given image.",
            "Share a concise interpretation of the image provided.",
            "Present a compact description of the photo’s key features.",
            "Relay a brief, clear account of the picture shown.",
            "Render a clear and concise summary of the photo.",
            "Write a terse but informative summary of the picture.",
            "Create a compact narrative representing the image presented."
        ]

        # Template for label-specific questions (can be dataset-specific)
        # For extensibility, different prompts can be added per dataset.
        self.label_question_template = (
            "Generate questions to verify if the object in the image corresponds to the label '{label}'. "
            "The questions should be answerable with 'yes' or 'no'. Focus on attributes, features, or functions "
            "that are characteristic of the label '{label}'."
        )

    def generate_general_questions(self, label: str, question_count: int = 2) -> List[str]:
        """
        Generate a list of general questions about the image.
        Args:
            label (str): The label associated with the image.
            question_count (int): Number of questions to generate.
        Returns:
            List[str]: List of generated questions.
        """
        questions = []
        # Randomly select templates for diversity
        for _ in range(question_count):
            template = random.choice(self.general_templates)
            questions.append(template)
        return questions

    def generate_label_specific_questions(self, label: str, question_count: int = 4) -> List[str]:
        """
        Generate label-specific questions based on the label, using prompt templates.
        Args:
            label (str): The label/class name for which questions are generated.
            question_count (int): Number of questions to generate.
        Returns:
            List[str]: List of generated label-specific questions.
        """
        questions = []
        # For each label, generate questions using the dataset-specific prompt template
        for _ in range(question_count):
            prompt = self.label_question_template.format(label=label)
            questions.append(prompt)
        return questions

    def generate_questions_for_label(
        self, label: str, num_general: int = 2, num_label_specific: int = 4
    ) -> List[str]:
        """
        Generate combined questions: general + label-specific.
        Args:
            label (str): Label/class name.
            num_general (int): Number of general questions.
            num_label_specific (int): Number of label-specific questions.
        Returns:
            List[str]: Combined list of questions ready for inference.
        """
        questions: List[str] = []
        questions.extend(self.generate_general_questions(label, num_general))
        questions.extend(self.generate_label_specific_questions(label, num_label_specific))
        return questions
