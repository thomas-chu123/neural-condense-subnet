import bittensor as bt
from ..constants import constants


def build_rate_limit(metagraph, config, tier=None):
    S = metagraph.S
    if config.whitelist_uids:
        whitelist_uids = [int(uid) for uid in config.miner.whitelist_uids.split(",")]
    else:
        whitelist_uids = [i for i in range(len(S)) if S[i] > constants.MIN_STAKE]
    bt.logging.info(f"Whitelist uids: {whitelist_uids}")

    selected_tier_config = constants.TIER_CONFIG[tier or config.miner.tier]
    rpe = selected_tier_config.requests_per_epoch

    # Calculate total stake of whitelisted UIDs
    total_stake = sum(S[uid] for uid in whitelist_uids)

    # Compute rate limits based on normalized stakes
    rate_limits = {}
    for uid in whitelist_uids:
        normalized_stake = S[uid] / total_stake if total_stake > 0 else 0
        rate_limits[uid] = max(int(rpe * normalized_stake), 10)

    # Set rate limit to 0 for non-whitelisted UIDs
    for uid in range(len(S)):
        if uid not in whitelist_uids:
            rate_limits[uid] = 0

    bt.logging.debug(f"Rate limits: {rate_limits}")
    return rate_limits
