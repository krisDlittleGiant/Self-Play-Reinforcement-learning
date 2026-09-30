"""Authoritative match logs and synchronous rollout collection."""

from self_play_grpo.rollouts.collector import MatchCollector
from self_play_grpo.rollouts.schema import MatchRecord, TurnRecord

__all__ = ["MatchCollector", "MatchRecord", "TurnRecord"]

