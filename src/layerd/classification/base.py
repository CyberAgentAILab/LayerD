"""Base class for element classification strategies."""

from abc import ABC, abstractmethod

from layerd.types import Element


class ElementLabeler(ABC):
    """Base class for element classification strategies.

    Provides a pluggable architecture for classifying elements by type.
    Subclasses implement specific classification logic (entropy, gradient, etc.).

    Example:
        >>> class MyLabeler(ElementLabeler):
        ...     def classify(self, elements):
        ...         # Custom classification logic
        ...         return elements
    """

    @abstractmethod
    def classify(self, elements: list[Element]) -> list[Element]:
        """Classify elements by modifying their 'type' field.

        Args:
            elements: List of elements to classify

        Returns:
            List of elements with updated 'type' fields
        """
        raise NotImplementedError
