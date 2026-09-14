"""
ml/business_impact.py
----------------------
Business-impact simulation (project spec Phase 16 / Question 2 item 10:
"The system must be evaluated on business metrics (customers retained,
revenue saved) not just technical metrics.")

Every dollar figure this module produces is a SIMULATION built from
business assumptions this project has no real campaign data to verify
(retention success rate, intervention cost, customer lifetime) -- not a
measured outcome. `avg_monthly_revenue` and the confusion matrix counts are
the only inputs computed from real data; everything else is a configurable
assumption with a clearly-labeled default. Nothing here should ever be
read as "the model saved $X" -- only as "if these assumptions hold, here
is what the model's test-set performance would imply." See
DEFAULT_ASSUMPTIONS below for exactly which numbers are assumptions, and
GET /business-impact in backend/main.py for how this is labeled wherever
it's surfaced.

Why simulate at all rather than skip it: the project spec explicitly
requires evaluating business impact, not just precision/recall/ROC-AUC,
and no real retention-campaign data exists yet to compute an actual
figure. A clearly-labeled simulation, adjustable to real numbers once they
exist, is more honest and more useful than silence -- as long as it is
never presented as a measured result.
"""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_ASSUMPTIONS = {
    # Fraction of TRUE churners who, when proactively contacted, are
    # actually retained by the intervention. 30% is a commonly-cited
    # ballpark in telecom retention-campaign literature -- NOT measured
    # from this company's actual campaigns (none exist yet; no
    # labeled-feedback loop exists, same limitation ml/retraining_graph.py
    # documents for `actual_outcome`).
    "retention_success_rate": 0.30,
    # Assumed cost (agent time + discount/offer) of one retention contact.
    "intervention_cost_per_contact": 15.0,
    # Assumed number of additional months a successfully-retained customer
    # stays -- i.e. how many months of revenue "saving them" is worth.
    "customer_lifetime_months": 12,
}


def compute_confusion_and_revenue(
    y_true: np.ndarray, y_pred: np.ndarray, monthly_charges: np.ndarray
) -> dict[str, Any]:
    """
    Extracts the two REAL, data-derived inputs the simulation needs: the
    confusion matrix (from true/predicted labels) and the average monthly
    revenue per customer (from actual MonthlyCharges values). No business
    assumptions are involved here -- this is pure measurement.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    confusion_matrix = {
        "true_positive": int(((y_pred == 1) & (y_true == 1)).sum()),
        "false_positive": int(((y_pred == 1) & (y_true == 0)).sum()),
        "false_negative": int(((y_pred == 0) & (y_true == 1)).sum()),
        "true_negative": int(((y_pred == 0) & (y_true == 0)).sum()),
    }
    return {
        "confusion_matrix": confusion_matrix,
        "avg_monthly_revenue": round(float(np.asarray(monthly_charges).mean()), 2),
    }


def _scenario(customers_flagged, retained, avg_monthly_revenue, cost_per_contact, lifetime_months) -> dict[str, Any]:
    revenue_saved = retained * avg_monthly_revenue * lifetime_months
    intervention_cost = customers_flagged * cost_per_contact
    roi = round(revenue_saved / intervention_cost, 2) if intervention_cost > 0 else None
    return {
        "customers_flagged": round(customers_flagged, 2),
        "expected_customers_retained": round(retained, 2),
        "estimated_revenue_saved": round(revenue_saved, 2),
        "estimated_intervention_cost": round(intervention_cost, 2),
        "estimated_net_benefit": round(revenue_saved - intervention_cost, 2),
        "roi": roi,
    }


def simulate_business_impact(
    confusion_matrix: dict[str, int],
    avg_monthly_revenue: float,
    retention_success_rate: float = DEFAULT_ASSUMPTIONS["retention_success_rate"],
    intervention_cost_per_contact: float = DEFAULT_ASSUMPTIONS["intervention_cost_per_contact"],
    customer_lifetime_months: int = DEFAULT_ASSUMPTIONS["customer_lifetime_months"],
) -> dict[str, Any]:
    """
    Simulates business impact from a confusion matrix + business
    assumptions, comparing three scenarios on identical footing:

    - model_targeted: contact only customers the model predicts CHURN.
    - contact_everyone: the traditional blanket-retention-campaign
      approach the model is meant to improve on (contact all customers).
    - do_nothing: no retention campaign at all (the "why bother" baseline).

    Only TRUE churners (true positives, or all actual churners for
    contact_everyone) can actually be "retained" by an intervention --
    contacting a customer who was never going to churn (a false positive)
    costs money but saves nothing.
    """
    tp = confusion_matrix["true_positive"]
    fp = confusion_matrix["false_positive"]
    fn = confusion_matrix["false_negative"]
    tn = confusion_matrix["true_negative"]

    total_customers = tp + fp + fn + tn
    actual_churners = tp + fn

    scenarios = {
        "model_targeted": _scenario(
            tp + fp, tp * retention_success_rate, avg_monthly_revenue,
            intervention_cost_per_contact, customer_lifetime_months,
        ),
        "contact_everyone": _scenario(
            total_customers, actual_churners * retention_success_rate, avg_monthly_revenue,
            intervention_cost_per_contact, customer_lifetime_months,
        ),
        "do_nothing": _scenario(0, 0, avg_monthly_revenue, intervention_cost_per_contact, customer_lifetime_months),
    }

    missed_revenue_at_risk = round(fn * avg_monthly_revenue * customer_lifetime_months, 2)

    return {
        "confusion_matrix": confusion_matrix,
        "assumptions": {
            "avg_monthly_revenue": round(avg_monthly_revenue, 2),
            "retention_success_rate": retention_success_rate,
            "intervention_cost_per_contact": intervention_cost_per_contact,
            "customer_lifetime_months": customer_lifetime_months,
        },
        "scenarios": scenarios,
        "missed_revenue_at_risk": missed_revenue_at_risk,
        "note": (
            "SIMULATED, not measured. retention_success_rate, "
            "intervention_cost_per_contact, and customer_lifetime_months are "
            "business assumptions this project has no real campaign data to "
            "verify -- adjust them to match actual figures once available. "
            "avg_monthly_revenue and the confusion matrix are the only inputs "
            "computed from real data. Compare scenarios by 'roi' (return per "
            "dollar spent), not just 'estimated_net_benefit' -- contacting "
            "every customer can show a higher raw net benefit simply because "
            "it reaches every true churner too, including the ones the model "
            "missed, while spending far more to get there; roi shows which "
            "approach uses the retention budget more efficiently."
        ),
    }


def scale_to_population(report: dict[str, Any], target_population: int, source_population: int) -> dict[str, Any]:
    """
    Proportionally projects a business-impact report computed on a smaller
    set (e.g. a ~1,400-row test set) to a larger hypothetical customer base
    (e.g. the project spec's "10 million customers"), assuming the same
    churn rate and prediction behavior hold at scale. That assumption is
    itself unverified -- stated plainly in the returned note, not
    something this project has evidence for one way or another.
    """
    if source_population <= 0:
        raise ValueError("source_population must be positive")
    factor = target_population / source_population

    scaled_scenarios = {
        name: {
            "customers_flagged": round(scenario["customers_flagged"] * factor, 1),
            "expected_customers_retained": round(scenario["expected_customers_retained"] * factor, 2),
            "estimated_revenue_saved": round(scenario["estimated_revenue_saved"] * factor, 2),
            "estimated_intervention_cost": round(scenario["estimated_intervention_cost"] * factor, 2),
            "estimated_net_benefit": round(scenario["estimated_net_benefit"] * factor, 2),
            "roi": scenario["roi"],  # a ratio -- unchanged by proportional scaling
        }
        for name, scenario in report["scenarios"].items()
    }

    return {
        "scale_factor": round(factor, 4),
        "target_population": target_population,
        "source_population": source_population,
        "confusion_matrix": {k: round(v * factor, 1) for k, v in report["confusion_matrix"].items()},
        "scenarios": scaled_scenarios,
        "note": (
            "Proportionally scaled from a smaller test set, assuming the same "
            "churn rate and prediction behavior hold at this population size -- "
            "an assumption, not a validated finding. Real deployment at this "
            "scale could behave differently."
        ),
    }
