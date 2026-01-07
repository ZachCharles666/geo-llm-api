from app.services.payments.models import NormalizedBillingEvent


def map_paddle_event_to_action(event: NormalizedBillingEvent) -> dict:
    """
    Map a normalized Paddle event into a billing action payload.

    TODO: Expand for additional Paddle event types once parsed.
    """
    action = None
    if event.event_type in ("subscription_activated", "subscription_canceled"):
        action = event.event_type
    elif event.event_type in ("subscription.activated", "subscription.canceled"):
        action = event.event_type.replace(".", "_")

    return {
        "action": action,
        "provider": event.provider,
        "event_id": event.event_id,
        "event_type": event.event_type,
        "occurred_at": event.occurred_at,
        "tier": event.tier,
        "status": event.status,
        "subscription_id": event.subscription_id,
        "customer_id": event.customer_id,
        "price_id": event.price_id,
    }
