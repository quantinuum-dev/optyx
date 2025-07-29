from abc import ABC, abstractmethod
from optyx.core.channel import Diagram
from cotengra import (
    ReusableHyperOptimizer,
    HyperOptimizer,
    HyperCompressedOptimizer,
    ReusableHyperCompressedOptimizer
)
import numpy as np

# contraction returns a channel box which then in then has
# methods to get probs etc...

class AbstractBackend(ABC):
    """
    Abstract base class for backend implementations.
    All backends must implement the `eval` method.
    """

    def _detect_return_type(self, diagram: Diagram):
        """
        Detect the return type of the diagram based on its domain and codomain.

        Single amplitudes, probability distributions, or tensors?

        Args:
            diagram (Diagram): The diagram to analyze.
        """

        if len(diagram.dom) == 0 and len(diagram.cod) == 0:
            return 'amplitude'
        elif len(diagram.dom) == 0 or len(diagram.cod) == 0:
            return 'distribution'
        else:
            return 'tensor'

    def _convert_array_to_prob_dist(self, result):
        """
        Convert a result array to a probability distribution.

        Args:
            result: The array to convert.

        Returns:
            A list of tuples of the form (occupation configuration, probability).
        """

        return {idx: val for idx, val in np.ndenumerate(result) if val != 0}

    def _convert_diagram_to_quimb(diagram: Diagram):
        if diagram.is_pure():
            return diagram.get_kraus().to_tensor().to_quimb()
        return diagram.double().to_tensor().to_quimb()

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

    def eval(self, diagram, hyperoptimiser=None, contraction_params=None):
        """
        Evaluate the diagram using Quimb.

        Args:
            diagram (Diagram): The diagram to evaluate.

        Returns:
            The result of the evaluation.
        """
        return_type = self._detect_return_type(diagram)
        quimb_tn = self._convert_diagram_to_quimb(diagram)
        if hyperoptimiser is None:
            return quimb_tn^...
        else:

            if isinstance(
                hyperoptimiser,
                ReusableHyperOptimizer | HyperOptimizer
            ):
                results = quimb_tn.contract(
                    optimize=hyperoptimiser,
                    **contraction_params
                )
            elif isinstance(
                hyperoptimiser,
                ReusableHyperCompressedOptimizer | HyperCompressedOptimizer
            ):
                results = quimb_tn.contract_compressed(
                    optimize=hyperoptimiser,
                    **contraction_params
                )
            else:
                raise ValueError(
                    "Unsupported hyperoptimiser type. Use ReusableHyperOptimizer, HyperOptimizer, ReusableHyperCompressedOptimizer, or HyperCompressedOptimizer."
                )

            if not isinstance(results, (complex, float, int)):
                results = results.data

        if return_type in ['amplitude', 'tensor']:
            return results
        elif return_type == 'distribution':
            return self._convert_array_to_prob_dist(results)
        else:
            raise ValueError(f"Unknown return type: {return_type}")


class PercevalBackend(AbstractBackend):
    """
    Backend implementation using Perceval.
    """

    def eval(self, diagram):
        """
        Evaluate the diagram using Perceval.

        Args:
            diagram (Diagram): The diagram to evaluate.

        Returns:
            The result of the evaluation.
        """
        #

        # Placeholder for Perceval evaluation logic
        pass  # Implement Perceval-specific evaluation logic here


