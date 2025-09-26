"""
Persistence factory for dependency injection (DIP).
"""

from .persistence import PersistenceStrategy, CacheManager, ConfigHasher, CacheKeyGenerator
from .file_persistence import FilePersistenceProvider


class PersistenceFactory:
    """Factory for creating persistence components (DIP - Dependency Injection)"""

    @staticmethod
    def create_file_persistence_strategy() -> PersistenceStrategy:
        """Create file-based persistence strategy"""
        # Create dependencies
        provider = FilePersistenceProvider()
        cache_manager = CacheManager(provider)
        hasher = ConfigHasher()
        key_generator = CacheKeyGenerator(hasher)

        # Inject dependencies
        return PersistenceStrategy(provider, cache_manager, key_generator)

    @staticmethod
    def create_s3_persistence_strategy(bucket_name: str) -> PersistenceStrategy:
        """Create S3-based persistence strategy (future extension)"""
        from .file_persistence import S3PersistenceProvider

        # Create dependencies
        provider = S3PersistenceProvider(bucket_name)
        cache_manager = CacheManager(provider)
        hasher = ConfigHasher()
        key_generator = CacheKeyGenerator(hasher)

        # Inject dependencies
        return PersistenceStrategy(provider, cache_manager, key_generator)