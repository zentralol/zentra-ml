# api/routers/runs.py
# Returns recent model training runs from the Supabase database.
# If Supabase isn't configured (no env vars set), it returns an empty list
# instead of crashing — the rest of the API works fine without it.

import logging
from fastapi import APIRouter, Query
from api.schemas import ModelRun, RunsResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/runs", tags=["runs"])


@router.get("/recent", response_model=RunsResponse)
def recent_runs(n: int = Query(20, ge=1, le=100, description="How many runs to return.")):
    # Try to connect to Supabase and fetch recent runs
    try:
        from scripts.supabase_client import ZentraSupabase
        sb = ZentraSupabase()
        df = sb.fetch_recent_runs(n=n)
    except EnvironmentError:
        # SUPABASE_URL or SUPABASE_KEY env vars not set — that's fine locally
        logger.info("Supabase not configured, returning empty run list.")
        return RunsResponse(runs=[], source="unavailable")
    except Exception:
        logger.exception("Failed to fetch runs from Supabase.")
        return RunsResponse(runs=[], source="unavailable")

    if df.empty:
        return RunsResponse(runs=[], source="supabase")

    # Convert dataframe rows to ModelRun objects
    records = df.to_dict(orient="records")
    runs = [ModelRun(**{k: v for k, v in r.items() if k in ModelRun.model_fields}) for r in records]
    return RunsResponse(runs=runs, source="supabase")
