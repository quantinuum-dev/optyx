from abc import ABC, abstractmethod

class AbstractBackend(ABC):
    """
    Abstract base class for backend implementations.
    All backends must implement the `eval` method.
    """

    @abstractmethod
    def eval(self, *args, **kwargs):
        """
        Evaluate the backend with the given arguments.

        Args:
            *args: Positional arguments for evaluation.
            **kwargs: Keyword arguments for evaluation.

        Returns:
            The result of the evaluation.
        """
        pass


class QuimbBackend(AbstractBackend):
    """
    Backend implementation using Quimb.
    """

    def eval(self, )