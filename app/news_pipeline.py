# news_pipeline.py
# Phase 1: Scheduled events + exchange status only
# No breaking news. No LLMs. No Twitter. No cleverness.

import math
from datetime import datetime
from typing import List, Optional
from news_schema import NewsObservation, EventScope


class ScheduledEvent:
    """
    Represents a known scheduled event (economic calendar).
    """

    def __init__(
        self,
        event_time: datetime,
        event_type: str,
        description: str,
        base_risk: float,
        scope: EventScope
    ):
        self.event_time = event_time
        self.event_type = event_type
        self.description = description
        self.base_risk = base_risk
        self.scope = scope


class NewsPipeline:
    """
    Contextual risk and regime modifier pipeline.

    Phase 1:
    - Scheduled macro events
    - Exchange operational status

    Risk-aware, not sentiment-aware.
    """

    def __init__(
        self,
        symbol: str,
        risk_window_hours: float = 2.0,
        decay_halflife_hours: float = 24.0
    ):
        self.symbol = symbol
        self.risk_window_hours = risk_window_hours
        self.decay_halflife_hours = decay_halflife_hours

        self.scheduled_events: List[ScheduledEvent] = []
        self.exchange_operational = True

    def add_scheduled_event(
        self,
        event_time: datetime,
        event_type: str,
        description: str,
        base_risk: float,
        scope: EventScope
    ) -> None:
        self.scheduled_events.append(
            ScheduledEvent(
                event_time,
                event_type,
                description,
                base_risk,
                scope
            )
        )

    def _compute_time_decay(self, event_time: datetime, now: datetime) -> float:
        """
        Half-life decay.
        50% decay every decay_halflife_hours.
        """
        hours_elapsed = (now - event_time).total_seconds() / 3600

        if hours_elapsed < 0:
            return 1.0

        decay = 0.5 ** (hours_elapsed / self.decay_halflife_hours)
        return max(0.0, min(1.0, decay))

    def _compute_event_risk(self, event: ScheduledEvent, now: datetime) -> float:
        """
        Risk ramps up before event, peaks at event, decays after.
        """
        hours_until = (event.event_time - now).total_seconds() / 3600

        if hours_until > self.risk_window_hours:
            return 0.0

        if hours_until >= 0:
            proximity = 1.0 - (hours_until / self.risk_window_hours)
            return event.base_risk * proximity

        time_decay = self._compute_time_decay(event.event_time, now)
        return event.base_risk * time_decay

    def _find_dominant_event(self, now: datetime) -> Optional[ScheduledEvent]:
        max_risk = 0.0
        dominant = None

        for event in self.scheduled_events:
            risk = self._compute_event_risk(event, now)
            if risk > max_risk:
                max_risk = risk
                dominant = event

        return dominant if max_risk > 0.1 else None

    def _compute_narrative_intensity(self, event_risk: float) -> float:
        return min(1.0, event_risk * 1.2)

    def observe(self, now: datetime) -> NewsObservation:
        """
        Produce a NewsObservation snapshot.
        """

        dominant_event = self._find_dominant_event(now)

        # Exchange shock applies regardless of events
        exchange_shock = not self.exchange_operational

        if dominant_event is None:
            return NewsObservation(
                time=now,
                symbol=self.symbol,
                event_risk=0.0,
                shock_flag=exchange_shock,
                narrative_intensity=0.0,
                time_decay=1.0,
                event_scope=EventScope.BTC_ONLY,
                confidence=1.0,
                is_sparse=False,
                event_type="none"
            )

        event_risk = self._compute_event_risk(dominant_event, now)
        time_decay = self._compute_time_decay(dominant_event.event_time, now)
        narrative_intensity = self._compute_narrative_intensity(event_risk)

        shock_flag = exchange_shock or event_risk > 0.9

        return NewsObservation(
            time=now,
            symbol=self.symbol,
            event_risk=event_risk,
            shock_flag=shock_flag,
            narrative_intensity=narrative_intensity,
            time_decay=time_decay,
            event_scope=dominant_event.scope,
            confidence=1.0,
            is_sparse=False,
            event_type=dominant_event.event_type
        )

    def set_exchange_status(self, operational: bool) -> None:
        self.exchange_operational = operational
