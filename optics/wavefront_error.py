from __future__ import annotations


class WavefrontError:
    """
    Base class for all wavefront errors.
    """

    def opd(
        self,
        x,
        y,
    ):
        """
        Returns optical path difference [m].
        """

        raise NotImplementedError

    def randomize(self):
        """
        Optional Monte-Carlo realization.
        """

        pass
