# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## architecture.py

```python
## architecture.py
from typing import List
import networkx as nx
import matplotlib.pyplot as plt
from components import Component
from typing import Optional
import os

class Architecture:
    """
    Represents a software architecture model, including its components, style, and relationships.
    Provides methods to construct, visualize, and manage the architectural structure.

    Attributes:
        name (str): Name or identifier for the architecture.
        components (List[Component]): List of components within the architecture.
        style_type (str): The architecture style, e.g., 'Microservices', 'Serverless', 'Event-Driven'.
    """

    def __init__(self, name: Optional[str] = None, style_type: str = "Microservices"):
        """
        Initializes the Architecture instance.

        Args:
            name (Optional[str]): Name of the architecture. Defaults to 'DefaultArchitecture'.
            style_type (str): The style of the architecture, defaulting to 'Microservices'.
        """
        self.name: str = name if name is not None else "DefaultArchitecture"
        self.components: List[Component] = []
        self.style_type: str = style_type

    def add_component(self, component: Component) -> None:
        """
        Adds a component to the architecture.

        Args:
            component (Component): The component instance to add.
        """
        if component not in self.components:
            self.components.append(component)

    def construct(self) -> None:
        """
        Constructs a sample architecture based on 'style_type' from configuration:
        - For 'Microservices': creates loosely coupled components with dependencies.
        - For 'Serverless': creates lightweight functions, minimal dependencies.
        - For 'Event-Driven': creates event producers/consumers with event flow dependencies.

        This method populates the components list accordingly.
        """
        self.components.clear()

        if self.style_type == "Microservices":
            # Example components for Microservices architecture
            order_service = Component("Order Service")
            payment_service = Component("Payment Service")
            notification_service = Component("Notification Service")

            # Dependencies: Order depends on Payment; Notification depends on Order
            order_service.add_dependency(payment_service)
            notification_service.add_dependency(order_service)

            self.add_component(order_service)
            self.add_component(payment_service)
            self.add_component(notification_service)

        elif self.style_type == "Serverless":
            # Example serverless functions/components
            auth_function = Component("Auth Function")
            payment_function = Component("Payment Function")
            email_function = Component("Email Function")
            # Minimal dependencies
            self.add_component(auth_function)
            self.add_component(payment_function)
            self.add_component(email_function)

        elif self.style_type == "Event-Driven":
            # Example event-based components
            product_event_producer = Component("ProductCreated Event Producer")
            order_event_consumer = Component("OrderCreated Event Consumer")
            shipping_event_producer = Component("ShippingStarted Event Producer")
            # Dependencies simulated as event flow
            order_event_consumer.add_dependency(product_event_producer)
            shipping_event_producer.add_dependency(order_event_consumer)

            self.add_component(product_event_producer)
            self.add_component(order_event_consumer)
            self.add_component(shipping_event_producer)

        else:
            # Default fallback architecture: empty or minimal
            default_component = Component("Default Component")
            self.add_component(default_component)

    def visualize(self, save_filepath: Optional[str] = None) -> None:
        """
        Visualizes the architecture as a directed graph,
        with nodes as components and edges representing dependencies.

        Args:
            save_filepath (Optional[str]): Path to save the visualization image.
                                            If None, the graph is displayed only.
        """
        graph = nx.DiGraph()

        # Add all components as nodes
        for comp in self.components:
            graph.add_node(comp.name)

        # Add edges based on dependencies
        for comp in self.components:
            for dep in comp.dependencies:
                # Add edge from component to its dependency
                if dep.name in graph.nodes:
                    graph.add_edge(comp.name, dep.name)

        # Draw the graph
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(graph)
        nx.draw_networkx_nodes(graph, pos, node_color='lightblue', node_size=1500)
        nx.draw_networkx_edges(graph, pos, arrowstyle='->', arrowsize=20)
        nx.draw_networkx_labels(graph, pos, font_size=12, font_weight='bold')

        plt.title(f"{self.style_type} Architecture: {self.name}")
        plt.axis('off')

        if save_filepath:
            # Save the visualization to file
            plt.savefig(save_filepath, format='png')
        else:
            # Display the visualization
            plt.show()

    def manage_relationships(self) -> None:
        """
        Ensures consistency and enforces constraints among components:
        - Checks for dependency cycles.
        - Validates that dependencies are among existing components.
        - Could enforce style-specific constraints.
        """
        # Check for dependency cycles
        graph = nx.DiGraph()
        for comp in self.components:
            for dep in comp.dependencies:
                if dep in self.components:
                    graph.add_edge(comp.name, dep.name)
        try:
            cycle = nx.find_cycle(graph, orientation='original')
            if cycle:
                print(f"Warning: Dependency cycle detected: {cycle}")
        except nx.exception.NetworkXNoCycle:
            # No cycles found
            pass

        # Additional constraints can be added here based on style_type
        # For example, microservices should have minimal dependencies
        # or serverless functions should be stateless, etc.
        # For now, this method is a placeholder for extension.

    def __repr__(self) -> str:
        """
        String representation of the Architecture object.
        """
        component_names = [comp.name for comp in self.components]
        return (f"Architecture(name='{self.name}', style_type='{self.style_type}', "
                f"components={component_names})")
```

## components.py

```python
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
```

## main.py

```python
## main.py
import yaml
import os
from components import Component
from principles import Principle
from architecture import Architecture
from visualization import draw_graph

def main():
    # Load configuration from 'config.yaml'
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Extract architecture styles from config
    architecture_styles = config.get('architecture_styles', [])
    principles_config = config.get('principles', [])
    
    # For simplicity, we'll demonstrate with the first architecture style specified
    if not architecture_styles:
        print("No architecture styles specified in configuration.")
        return

    style_entry = architecture_styles[0]
    style_name: str = style_entry.get('style_name', 'Microservices')
    style_description: str = style_entry.get('description', '')

    # Instantiate Architecture
    architecture_name = f"{style_name} Architecture Demo"
    architecture = Architecture(name=architecture_name, style_type=style_name)

    # Construct architecture components based on style
    architecture.construct()

    # Create and add components (as per constructed architecture)
    # Components are already created inside construct()

    # Instantiate principles based on configuration
    principles = []
    for p_cfg in principles_config:
        principle_name = p_cfg.get('name')
        # Description can be extended; for now, simple placeholder
        description = f"Enforces {principle_name}"
        principle = Principle(name=principle_name, description=description)
        principles.append(principle)

    # Apply each principle to each component
    for component in architecture.components:
        for principle in principles:
            # Apply principle, which returns boolean if compliant
            compliant = component.apply_principle(principle)
            # Log the application
            print(f"Component '{component.name}': Applied Principle '{principle.name}', Compliant: {compliant}")

    # Visualize architecture
    # Decide on output file path based on architecture style
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, f"{style_name}_architecture.png")
    print(f"Visualizing architecture and saving to {output_filepath}")
    draw_graph(architecture, filepath=output_filepath, show=False)

    # Optional: Could perform validation or add annotations
    # For example, highlight compliant vs non-compliant components
    # Here, we skip detailed validation for simplicity

    print("Demo completed. Architecture visualization saved successfully.")

if __name__ == "__main__":
    main()
```

## principles.py

```python
## principles.py
from typing import Optional
from components import Component

class Principle:
    """
    Represents a software design principle, such as Single Responsibility,
    Open-Closed, Interface Segregation, or Dependency Inversion.

    Attributes:
        name (str): Name of the principle.
        description (str): Description explaining the principle.
    """

    def __init__(self, name: str, description: Optional[str] = None):
        """
        Initializes a Principle with a name and optional description.

        Args:
            name (str): The name of the principle.
            description (Optional[str]): The detailed explanation of the principle.
        """
        self.name: str = name
        self.description: str = description if description is not None else ""

    def enforce(self, component: Component) -> bool:
        """
        Checks whether the given component complies with this principle.

        This implementation provides simplified placeholder logic
        tailored for demonstration purposes.

        Args:
            component (Component): The component to evaluate.

        Returns:
            bool: True if the component adheres to the principle, False otherwise.
        """
        # Placeholder enforcement logic based on principle name
        if self.name == "Single Responsibility Principle":
            # Assume a component with a 'purpose' attribute indicating its responsibility
            # Since components may lack this attribute, simulate with name check
            # For demonstration, assume components with names containing 'Service' are responsible for single purpose
            return "Service" in component.name
        elif self.name == "Open-Closed Principle":
            # Assume components designed with extension points end with '_ext'
            # For demo, check if component name ends with '_ext'
            return component.name.endswith('_ext')
        elif self.name == "Interface Segregation Principle":
            # Assume components implement interfaces; here simulate by number of dependencies
            # Components with dependencies less than or equal to 2 are adhering
            return len(component.dependencies) <= 2
        elif self.name == "Dependency Inversion Principle":
            # Assume components depend on abstractions if dependencies are on 'interface' components
            # For demo, if dependencies have names containing 'Interface', assume adherence
            return all("Interface" in dep.name for dep in component.dependencies)
        else:
            # For unknown principles, default to True
            return True
```

## visualization.py

```python
## visualization.py
import networkx as nx
import matplotlib.pyplot as plt
from architecture import Architecture
from components import Component
from typing import Optional, Dict, Any

def draw_graph(
    architecture: Architecture,
    filepath: Optional[str] = None,
    show: bool = True,
    highlight_principles: Optional[Dict[str, Any]] = None
) -> nx.DiGraph:
    """
    Visualizes the given architecture as a directed graph.

    Args:
        architecture (Architecture): The architecture instance to visualize.
        filepath (Optional[str]): Path to save the visualization image. Defaults to None.
        show (bool): Whether to display the graph interactively. Defaults to True.
        highlight_principles (Optional[Dict[str, Any]]): Optional dict specifying component names
            mapped to their principle compliance info (e.g., {'ComponentA': {'compliant': True}}).
            This can be used to color nodes based on principles adherence.

    Returns:
        nx.DiGraph: The generated NetworkX directed graph object.
    """
    # Initialize directed graph
    graph = nx.DiGraph()

    # Create a mapping for component names to component objects for validation
    name_to_component = {
        comp.name: comp for comp in architecture.components
    }

    # Add nodes with attributes
    for comp in architecture.components:
        # Decide node color based on principle compliance if info provided
        color = 'lightblue'  # default color
        if highlight_principles and comp.name in highlight_principles:
            comp_info = highlight_principles[comp.name]
            # For demonstration, use 'compliant' bool to assign color
            if comp_info.get('compliant', True):
                color = 'lightgreen'
            else:
                color = 'salmon'
        # Add node with label and color attribute
        graph.add_node(comp.name, label=comp.name, color=color)

    # Add edges based on dependencies
    for comp in architecture.components:
        for dep in comp.dependencies:
            # Verify that the dependency is part of the architecture
            if dep.name in name_to_component:
                graph.add_edge(comp.name, dep.name)

    # Prepare for visualization
    # Compute layout
    pos = nx.spring_layout(graph, seed=42)

    # Extract node colors
    node_colors = [graph.nodes[node].get('color', 'lightblue') for node in graph.nodes()]

    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=1500, edgecolors='black')

    # Draw edges
    nx.draw_networkx_edges(graph, pos, arrowstyle='->', arrowsize=20, linewidths=1)

    # Draw labels
    labels = {node: node for node in graph.nodes()}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10, font_weight='bold')

    # Add title
    plt.title(f"{architecture.style_type} Architecture: {architecture.name}")

    # Remove axes
    plt.axis('off')

    # Save figure if filepath provided
    if filepath:
        plt.savefig(filepath, format='png', bbox_inches='tight')

    # Display plot if 'show' is True
    if show:
        plt.show()
    else:
        plt.close()

    return graph
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\outputs\paper2code\SoftwareArchitectureandSoftwareDesign\SoftwareArchitectureandSoftwareDesign_repo`
