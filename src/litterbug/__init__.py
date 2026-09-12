"""Trash detection and instance segmentation on a conveyor belt.

One subpackage per concern: ``common`` holds the shared contract, configuration and
runtime guards; ``training`` and ``running`` do the work; ``cli`` parses arguments and
dispatches.

``training`` and ``running`` never import each other. Anything they genuinely share
belongs in ``common``.
"""
