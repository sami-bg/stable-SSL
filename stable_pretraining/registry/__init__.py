# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Local SQLite-based run registry for fast sweep/run querying.

Provides a Lightning Logger (:class:`RegistryLogger`) that transparently
captures ``self.log()`` calls into a SQLite database, and a query API
(:func:`open_registry`) to retrieve runs, hparams, and summary stats.
"""

from .logger import RegistryLogger
from .query import Registry, RunRecord, open_registry

__all__ = ["RegistryLogger", "Registry", "RunRecord", "open_registry"]
