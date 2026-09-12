"""Shared surface only: the class contract, configuration and runtime guards.

Nothing here may depend on ``training`` or ``running`` — those two must never need
each other, and this package is what keeps that true.
"""
