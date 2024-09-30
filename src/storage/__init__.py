"""
Storage Factory class
"""


class Storage:
    """
    Storage Factory class
    """

    def __init__(self, storage_type, embedding_function):
        self.storage_type = storage_type
        self.embedding_function = embedding_function

    def get_storage(self):
        """
        Get the storage object based on the storage type
        """
        if self.storage_type == "faiss":
            from .faiss import FaissStorage  # pylint: disable=import-outside-toplevel

            return FaissStorage(self.embedding_function)
        else:
            raise NotImplementedError(
                f"Storage type {self.storage_type} not implemented"
            )
