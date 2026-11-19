"""Everything that touches OpenGL lives in this package and nowhere else.

Simulation packages (core, world, entities, ai, animation, levels) are rendered *by* this
package; they never import it. See tests/test_architecture.py for the enforced rule.

PyOpenGL's per-call ``glGetError`` round-trip is disabled here unless ``CYBERDUKE_GL_DEBUG``
is set; this must happen before ``OpenGL.GL`` is imported for the first time.
"""

import os

import OpenGL

GL_DEBUG = bool(os.environ.get("CYBERDUKE_GL_DEBUG"))
OpenGL.ERROR_CHECKING = GL_DEBUG
