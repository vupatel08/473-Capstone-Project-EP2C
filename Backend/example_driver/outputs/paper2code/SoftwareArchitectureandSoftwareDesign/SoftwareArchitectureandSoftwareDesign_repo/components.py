## components.py
from typing import List, Optional

class Component:
    """
    Represents a modular unit within a software architecture or design.

    Attributes:
        name (str): Unique identifier for the component.
        dependencies (List[Component]): List of other components this component depends on.
        applied_principles (List[Principle]): List of principles this component adheres to or has been tested against.
    """

    def __init__(self, name: str, dependencies: Optional[List['Component']] = None):
        """
        Initializes a Component with a name and optional list of dependencies.

        Args:
            name (str): Name of the component.
            dependencies (Optional[List[Component]]): Dependencies of the component. Defaults to empty list.
        """
        self.name: str = name
        self.dependencies: List['Component'] = dependencies if dependencies is not None else []
        self.applied_principles: List['Principle'] = []

    def add_dependency(self, component: 'Component') -> None:
        """
        Adds a dependency to this component.

        Args:
            component (Component): The component to add as a dependency.
        """
        if component not in self.dependencies:
            self.dependencies.append(component)

    def apply_principle(self, principle: 'Principle') -> bool:
        """
        Checks whether this component complies with the given principle.

        Args:
            principle (Principle): The principle to validate against this component.

        Returns:
            bool: True if the component adheres to the principle, False otherwise.
        """
        is_compliant = principle.enforce(self)
        if is_compliant:
            self.applied_principles.append(principle)
        return is_compliant

    def __repr__(self) -> str:
        """
        Returns a string representation of the component, including its dependencies.

        Returns:
            str: String representation.
        """
        dep_names = [dep.name for dep in self.dependencies]
        return f"Component(name='{self.name}', dependencies={dep_names})"
