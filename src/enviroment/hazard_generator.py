"""
src/environment/hazard_generator.py
====================================
Stochastic hazard generation for DisasterResponseBenchmark.

Implements the transition function P(s' | s, a) described in Section 4.1
of the paper:

  "The transition function P is implemented via a stochastic hazard
   propagation model in which storm and flood risk fields evolve according
   to spatially correlated Gaussian processes conditioned on sampled
   hazard events."

Hazard events arrive according to a Poisson process (λ = 0.05 per step).
Each event is one of: storm, flood, or combined.
Risk intensity fields are propagated via Gaussian spatial diffusion on the
20×20 grid and decay exponentially over time.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HazardEvent:
    """A single hazard event sampled at a given timestep."""
    event_type: str          # "storm" | "flood" | "combined"
    origin: Tuple[int, int]  # (row, col) on the grid
    intensity: float         # initial intensity in [0, 1]
    timestep: int            # simulation step at which it was generated


@dataclass
class HazardState:
    """Full hazard state returned each timestep."""
    storm_field: np.ndarray        # (H, W) storm risk intensity [0, 1]
    flood_field: np.ndarray        # (H, W) flood risk intensity [0, 1]
    storm_probability: float       # scalar P(storm) in [0, 1]
    rainfall_intensity: float      # scalar normalised rainfall in [0, 1]
    river_level: float             # scalar river-level indicator in [0, 1]
    vulnerability: float           # scalar regional vulnerability in [0, 1]
    active_events: List[HazardEvent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class HazardGenerator:
    """
    Generates spatially correlated hazard fields for each simulation step.

    Parameters
    ----------
    grid_size : int
        Side length of the square grid (default 20).
    hazard_lambda : float
        Poisson arrival rate λ for new hazard events per step (default 0.05).
    gp_lengthscale : float
        Spatial correlation length-scale for Gaussian diffusion (grid cells).
    gp_variance : float
        Output variance of the Gaussian process kernel.
    decay_rate : float
        Exponential decay rate applied to field intensities each step.
    seed : int | None
        Random seed for reproducibility.
    """

    HAZARD_TYPES: List[str] = ["storm", "flood", "combined"]

    def __init__(
        self,
        grid_size: int = 20,
        hazard_lambda: float = 0.05,
        gp_lengthscale: float = 3.0,
        gp_variance: float = 1.0,
        decay_rate: float = 0.05,
        seed: int | None = None,
    ) -> None:
        self.grid_size = grid_size
        self.hazard_lambda = hazard_lambda
        self.gp_lengthscale = gp_lengthscale
        self.gp_variance = gp_variance
        self.decay_rate = decay_rate

        self.rng = np.random.default_rng(seed)
        self._timestep: int = 0

        # Persistent risk fields — decay and accumulate across steps
        self._storm_field: np.ndarray = np.zeros((grid_size, grid_size))
        self._flood_field: np.ndarray = np.zeros((grid_size, grid_size))

        # Regional vulnerability is a fixed spatial map sampled once at init
        self._vulnerability_map: np.ndarray = self._init_vulnerability_map()

        # Track all active events for observation routing
        self._active_events: List[HazardEvent] = []

        # Pre-compute Gaussian kernel for spatial diffusion
        self._kernel = self._build_gaussian_kernel(radius=5)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> HazardState:
        """Reset all hazard fields to zero and return the initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._timestep = 0
        self._storm_field = np.zeros((self.grid_size, self.grid_size))
        self._flood_field = np.zeros((self.grid_size, self.grid_size))
        self._active_events = []
        self._vulnerability_map = self._init_vulnerability_map()

        return self._build_state()

    def step(self) -> HazardState:
        """
        Advance the hazard model by one timestep.

        Steps:
        1. Decay existing fields.
        2. Sample new events via Poisson process.
        3. Inject new event intensities at origin cells.
        4. Diffuse fields via Gaussian spatial smoothing.
        5. Clip and return the updated HazardState.
        """
        self._timestep += 1

        # 1. Exponential decay of existing risk
        self._storm_field *= (1.0 - self.decay_rate)
        self._flood_field *= (1.0 - self.decay_rate)

        # 2. Sample number of new events this step
        n_events = self.rng.poisson(self.hazard_lambda)

        # 3. Generate each new event and inject into the appropriate field
        new_events: List[HazardEvent] = []
        for _ in range(n_events):
            event = self._sample_event()
            new_events.append(event)
            r, c = event.origin

            if event.event_type in ("storm", "combined"):
                self._storm_field[r, c] = min(
                    1.0, self._storm_field[r, c] + event.intensity
                )
            if event.event_type in ("flood", "combined"):
                self._flood_field[r, c] = min(
                    1.0, self._flood_field[r, c] + event.intensity
                )

        # 4. Gaussian spatial diffusion (models risk propagation)
        self._storm_field = self._diffuse(self._storm_field)
        self._flood_field = self._diffuse(self._flood_field)

        # Update active events (keep only recent ones)
        self._active_events = new_events  # simplification: only current step

        return self._build_state()

    @property
    def timestep(self) -> int:
        return self._timestep

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _sample_event(self) -> HazardEvent:
        """Sample a single hazard event uniformly over the grid."""
        event_type = self.rng.choice(self.HAZARD_TYPES)
        origin = (
            int(self.rng.integers(0, self.grid_size)),
            int(self.rng.integers(0, self.grid_size)),
        )
        intensity = float(self.rng.uniform(0.3, 1.0))
        return HazardEvent(
            event_type=event_type,
            origin=origin,
            intensity=intensity,
            timestep=self._timestep,
        )

    def _diffuse(self, field: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian spatial diffusion via separable 2-D convolution.
        Uses manual correlation to avoid scipy dependency.
        """
        from scipy.ndimage import convolve
        diffused = convolve(field, self._kernel, mode="reflect")
        return np.clip(diffused, 0.0, 1.0)

    def _build_gaussian_kernel(self, radius: int = 5) -> np.ndarray:
        """Build a normalised 2-D Gaussian kernel for spatial diffusion."""
        size = 2 * radius + 1
        ax = np.arange(-radius, radius + 1, dtype=np.float64)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(
            -(xx ** 2 + yy ** 2) / (2.0 * self.gp_lengthscale ** 2)
        )
        kernel /= kernel.sum()
        return kernel.astype(np.float32)

    def _init_vulnerability_map(self) -> np.ndarray:
        """
        Generate a fixed regional vulnerability map.
        Sampled once per episode; reflects persistent infrastructure exposure.
        Values in [0, 1].
        """
        base = self.rng.uniform(0.1, 0.9, size=(self.grid_size, self.grid_size))
        # Smooth the vulnerability map for spatial coherence
        from scipy.ndimage import gaussian_filter
        smoothed = gaussian_filter(base.astype(np.float32), sigma=2.0)
        return np.clip(smoothed, 0.0, 1.0)

    def _build_state(self) -> HazardState:
        """Assemble scalar summaries and spatial fields into a HazardState."""
        # Scalar storm probability: max field intensity as a proxy
        storm_prob = float(np.max(self._storm_field))

        # Scalar rainfall: mean of flood field (rainfall feeds flood risk)
        rainfall = float(np.mean(self._flood_field))

        # River level: weighted combination of flood field near the grid centre
        cx, cy = self.grid_size // 2, self.grid_size // 2
        river_region = self._flood_field[
            max(0, cx - 3) : cx + 3, max(0, cy - 3) : cy + 3
        ]
        river_level = float(np.mean(river_region))

        # Regional vulnerability: mean vulnerability across the grid
        vulnerability = float(np.mean(self._vulnerability_map))

        return HazardState(
            storm_field=self._storm_field.copy(),
            flood_field=self._flood_field.copy(),
            storm_probability=np.clip(storm_prob, 0.0, 1.0),
            rainfall_intensity=np.clip(rainfall, 0.0, 1.0),
            river_level=np.clip(river_level, 0.0, 1.0),
            vulnerability=np.clip(vulnerability, 0.0, 1.0),
            active_events=list(self._active_events),
        )

    def get_flat_state(self) -> np.ndarray:
        """
        Return a flat 24-dimensional state vector for synthetic training mode.
        Matches state_dim=24 from environment.yaml (Table 1 of the paper).

        Layout (24 dims):
          [0]      storm_probability
          [1]      rainfall_intensity
          [2]      river_level
          [3]      vulnerability
          [4..11]  8 storm field summary statistics
          [12..19] 8 flood field summary statistics
          [20..23] hazard event counts by type (storm, flood, combined, total)
        """
        hs = self._build_state()

        def field_stats(f: np.ndarray) -> np.ndarray:
            """Return 8 summary statistics for a 2-D field."""
            return np.array([
                f.mean(), f.max(), f.min(), f.std(),
                np.percentile(f, 25), np.percentile(f, 50),
                np.percentile(f, 75), (f > 0.5).mean(),
            ], dtype=np.float32)

        n_storm = sum(
            1 for e in hs.active_events if e.event_type in ("storm", "combined")
        )
        n_flood = sum(
            1 for e in hs.active_events if e.event_type in ("flood", "combined")
        )
        n_combined = sum(
            1 for e in hs.active_events if e.event_type == "combined"
        )
        n_total = len(hs.active_events)

        state = np.concatenate([
            [hs.storm_probability, hs.rainfall_intensity,
             hs.river_level, hs.vulnerability],         # 4
            field_stats(hs.storm_field),                # 8
            field_stats(hs.flood_field),                # 8
            [n_storm, n_flood, n_combined, n_total],    # 4
        ]).astype(np.float32)

        assert state.shape == (24,), f"Expected state_dim=24, got {state.shape}"
        return state
