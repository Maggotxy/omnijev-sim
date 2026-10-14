"""Choose among model commands without reading any external-world state."""

def choose_motor(now, reflex, brain, reflex_enabled, has_brain):
    if not has_brain:
        action = reflex.action(now) if reflex_enabled else "continue"
        record = reflex.record if reflex_enabled and action != "watchdog_hold" else None
        return action, record.get("speed_scale", 1) if record else 1, ("cerebellum" if record else "software" if action == "watchdog_hold" else "initial_intention"), record
    local_action = reflex.action(now) if reflex_enabled else "watchdog_hold"
    brain_action = brain.action(now)
    local = reflex.record if reflex_enabled and local_action != "watchdog_hold" else None
    cortical = brain.record if brain_action != "watchdog_hold" else None
    if local and local_action in ("brake", "retreat", "lift"):
        return local_action, local.get("speed_scale", 1), "cerebellum", local
    if local and local_action == "continue" and local.get("model_intervention") and local.get("reflex_id") == "continue" and cortical and local["observed_wall_s"] > cortical["observed_wall_s"]+.1:
        return "continue", local.get("speed_scale", 1), "cerebellum", local
    if cortical and brain_action != "continue":
        return brain_action, cortical.get("speed_scale", 1), "brain", cortical
    if local:
        return local_action, local.get("speed_scale", 1), "cerebellum", local
    if cortical:
        return brain_action, cortical.get("speed_scale", 1), "brain", cortical
    if now < reflex.command_ttl:
        return "continue", 1, "initial_intention", None
    return "watchdog_hold", 1, "software", None
