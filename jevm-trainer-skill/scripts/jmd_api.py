"""Shared API client and validation utils for JEV+MDP trainer skill."""

import json
import time

import requests


class JMDError(Exception):
    """Raised when a JEV+MDP API call fails."""


class JMDClient:
    """Thin client for the JEV+MDP HTTP backend."""

    def __init__(self, base_url="http://localhost:8000", timeout=120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _raise(self, resp: requests.Response) -> None:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise JMDError(f"HTTP {resp.status_code}: {detail}")

    def health(self) -> dict:
        resp = requests.get(self._url("/api/v1/health"), timeout=self.timeout)
        if not resp.ok:
            self._raise(resp)
        return resp.json()

    def predict(self, state_text: str, actions: list[str]) -> dict:
        payload = {"state_text": state_text, "actions": actions}
        resp = requests.post(self._url("/api/v1/predict"), json=payload, timeout=self.timeout)
        if not resp.ok:
            self._raise(resp)
        return resp.json()

    def submit_feedback(
        self,
        state_text: str,
        actions: list[str],
        optimal_action: int,
        transition_probs: list[list[float]],
        expected_returns: list[float],
    ) -> dict:
        payload = {
            "state_text": state_text,
            "actions": actions,
            "optimal_action": optimal_action,
            "transition_probs": transition_probs,
            "expected_returns": expected_returns,
        }
        resp = requests.post(self._url("/api/v1/feedback"), json=payload, timeout=self.timeout)
        if not resp.ok:
            self._raise(resp)
        return resp.json()

    def stats(self) -> dict:
        resp = requests.get(self._url("/api/v1/feedback/stats"), timeout=self.timeout)
        if not resp.ok:
            self._raise(resp)
        return resp.json()

    def training_records(self, limit: int = 20, offset: int = 0) -> list[dict]:
        resp = requests.get(
            self._url("/api/v1/admin/training-records"),
            params={"limit": limit, "offset": offset},
            timeout=self.timeout,
        )
        if not resp.ok:
            self._raise(resp)
        data = resp.json()
        if isinstance(data, dict):
            return data.get("records", data.get("items", []))
        return data if isinstance(data, list) else []

    def trigger_training(self) -> dict:
        resp = requests.post(self._url("/api/v1/admin/training/trigger"), timeout=self.timeout)
        if not resp.ok:
            self._raise(resp)
        return resp.json()

    def wait_for_training(self, poll_interval: float = 5.0, max_wait: float = 600.0) -> dict:
        """Poll training records until the most recent one leaves running/queued.
        Returns the newest completed/failed record dict, or None on timeout.
        """
        deadline = time.time() + max_wait
        last = None
        while time.time() < deadline:
            records = self.training_records(limit=1, offset=0)
            if records:
                newest = records[0]
                if newest.get("status") in ("completed", "failed", "cancelled"):
                    return newest
                last = newest
            time.sleep(poll_interval)
        return last


def load_scenarios(path: str | None, fallback: list[dict]) -> list[dict]:
    """Load scenarios from a JSON file; fall back to the provided defaults."""
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return fallback


def to_json(obj, indent: int = 2) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=indent)
