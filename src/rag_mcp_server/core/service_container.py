"""
Service Container - Dependency Inversion Principle implementation.

This container manages dependencies and their lifecycles, allowing
for easy testing and extensibility.
"""

from typing import Dict, Any, Callable, TypeVar, Type
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServiceContainer:
    """
    Dependency injection container following Dependency Inversion Principle.

    Allows registering services as:
    - Singletons (created once, reused)
    - Transients (new instance each time)
    - Factories (custom creation logic)
    """

    def __init__(self):
        """Initialize empty container."""
        self._singletons: Dict[Type, Any] = {}
        self._transients: Dict[Type, Callable] = {}
        self._factories: Dict[Type, Callable] = {}
        self._instances: Dict[Type, Any] = {}

    def register_singleton(self, interface: Type[T], implementation: Type[T]) -> None:
        """Register a singleton service."""
        self._singletons[interface] = implementation
        logger.debug(f"Registered singleton: {interface.__name__} -> {implementation.__name__}")

    def register_transient(self, interface: Type[T], implementation: Type[T]) -> None:
        """Register a transient service."""
        self._transients[interface] = implementation
        logger.debug(f"Registered transient: {interface.__name__} -> {implementation.__name__}")

    def register_factory(self, interface: Type[T], factory: Callable[[], T]) -> None:
        """Register a factory function."""
        self._factories[interface] = factory
        logger.debug(f"Registered factory for: {interface.__name__}")

    def register_instance(self, interface: Type[T], instance: T) -> None:
        """Register a pre-created instance."""
        self._instances[interface] = instance
        logger.debug(f"Registered instance: {interface.__name__}")

    def resolve(self, interface: Type[T]) -> T:
        """
        Resolve a service by its interface.

        Resolution order:
        1. Pre-created instances
        2. Singletons (create if not exists)
        3. Factories
        4. Transients

        Args:
            interface: The interface/type to resolve

        Returns:
            Instance of the requested type

        Raises:
            ValueError: If service is not registered
        """
        # Check for pre-created instances
        if interface in self._instances:
            return self._instances[interface]

        # Check for singletons
        if interface in self._singletons:
            if interface not in self._instances:
                implementation = self._singletons[interface]
                self._instances[interface] = self._create_instance(implementation)
            return self._instances[interface]

        # Check for factories
        if interface in self._factories:
            return self._factories[interface]()

        # Check for transients
        if interface in self._transients:
            implementation = self._transients[interface]
            return self._create_instance(implementation)

        raise ValueError(f"Service not registered: {interface.__name__}")

    def _create_instance(self, implementation_class: Type[T]) -> T:
        """Create instance with basic dependency injection."""
        try:
            # Try to create without arguments first
            return implementation_class()
        except TypeError:
            # If that fails, try to inject known dependencies
            # This is a simplified DI - in a real system you'd use more sophisticated injection
            import inspect
            signature = inspect.signature(implementation_class.__init__)
            kwargs = {}

            for param_name, param in signature.parameters.items():
                if param_name == 'self':
                    continue

                param_type = param.annotation
                if param_type != param.empty and param_type in self._get_all_registered_types():
                    try:
                        kwargs[param_name] = self.resolve(param_type)
                    except ValueError:
                        # Parameter can't be resolved, skip it
                        pass

            return implementation_class(**kwargs)

    def _get_all_registered_types(self) -> set:
        """Get all registered interface types."""
        all_types = set()
        all_types.update(self._instances.keys())
        all_types.update(self._singletons.keys())
        all_types.update(self._factories.keys())
        all_types.update(self._transients.keys())
        return all_types

    def clear(self) -> None:
        """Clear all registrations and instances."""
        self._singletons.clear()
        self._transients.clear()
        self._factories.clear()
        self._instances.clear()
        logger.debug("Service container cleared")

    def is_registered(self, interface: Type[T]) -> bool:
        """Check if a service is registered."""
        return (interface in self._instances or
                interface in self._singletons or
                interface in self._factories or
                interface in self._transients)

    def get_registration_info(self) -> Dict[str, Any]:
        """Get information about registered services."""
        return {
            "singletons": [cls.__name__ for cls in self._singletons.keys()],
            "transients": [cls.__name__ for cls in self._transients.keys()],
            "factories": [cls.__name__ for cls in self._factories.keys()],
            "instances": [cls.__name__ for cls in self._instances.keys()],
            "total_registrations": len(self._get_all_registered_types())
        }


# Global service container instance
service_container = ServiceContainer()