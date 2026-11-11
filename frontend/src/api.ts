/** Typed fetch wrapper for the backend API (proxied via Vite in dev). */

import type {
  EarlyWarningResponse,
  EfficientFrontierResponse,
  GenerateDataResponse,
  HealthResponse,
  PortfolioAnalyticsResponse,
  RegimeForecastResponse,
  ReverseStressResponse,
  ScenarioInfo,
  StressTestRequest,
  StressTestResponse,
  TrainRegimesResponse,
} from "./types";

const BASE = "/api";

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
      else if (body.detail) detail = JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body — keep statusText */
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

export const api = {
  generateData: (nDays: number, seed: number): Promise<GenerateDataResponse> =>
    request("/generate-data", {
      method: "POST",
      body: JSON.stringify({ n_days: nDays, seed }),
    }),

  trainRegimes: (nRegimes = 3, seed = 7): Promise<TrainRegimesResponse> =>
    request("/train-regimes", {
      method: "POST",
      body: JSON.stringify({ n_regimes: nRegimes, seed }),
    }),

  stressTest: (payload: StressTestRequest): Promise<StressTestResponse> =>
    request("/stress-test", { method: "POST", body: JSON.stringify(payload) }),

  scenarios: (): Promise<{ scenarios: ScenarioInfo[] }> => request("/scenarios"),

  portfolioAnalytics: (
    weights: Record<string, number>,
  ): Promise<PortfolioAnalyticsResponse> =>
    request("/portfolio-analytics", {
      method: "POST",
      body: JSON.stringify({ weights }),
    }),

  efficientFrontier: (nSamples = 3000, seed = 11): Promise<EfficientFrontierResponse> =>
    request("/efficient-frontier", {
      method: "POST",
      body: JSON.stringify({ n_samples: nSamples, seed }),
    }),

  regimeForecast: (horizonDays = 126): Promise<RegimeForecastResponse> =>
    request(`/regime-forecast?horizon_days=${horizonDays}`),

  reverseStress: (
    weights: Record<string, number>,
    targetLoss: number,
    horizonDays = 21,
  ): Promise<ReverseStressResponse> =>
    request("/reverse-stress", {
      method: "POST",
      body: JSON.stringify({
        weights,
        target_loss: targetLoss,
        horizon_days: horizonDays,
      }),
    }),

  earlyWarning: (): Promise<EarlyWarningResponse> => request("/early-warning"),

  health: (): Promise<HealthResponse> => request("/health"),
};
